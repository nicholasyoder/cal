"""Tools exposed to the model. File access is confined to the workspace; web access is GET-only
and refuses private / loopback addresses so the agent can't poke at the local network."""

import inspect
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from ollama._utils import convert_function_to_tool

from .config import Config
from .memory import MemoryStore


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, cfg: Config, memory: MemoryStore):
        self.cfg = cfg.tools
        self.root = cfg.workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory = memory
        self.funcs = {
            f.__name__: f
            for f in [
                self.list_files, self.read_file, self.write_file, self.edit_file,
                self.delete_file, self.web_search, self.fetch_url, self.recall,
            ]
        }
        self.schemas = [_schema(f) for f in self.funcs.values()]

    def call(self, name: str, args: dict) -> str:
        f = self.funcs.get(name)
        if f is None:
            return f"Error: no tool named {name!r}. Available: {', '.join(self.funcs)}"
        try:
            return f(**(args or {}))
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _path(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if not p.is_relative_to(self.root):
            raise ToolError(f"{path} is outside your workspace")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    # Files

    def list_files(self, path: str = ".") -> str:
        """List files and directories in your workspace.

        Args:
            path: Directory relative to the workspace root.
        """
        d = self._path(path)
        if not d.is_dir():
            raise ToolError(f"{path} is not a directory")
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"{self._rel(d)}/ is empty"
        return "\n".join(
            f"{self._rel(p)}/" if p.is_dir() else f"{self._rel(p)}  ({p.stat().st_size} bytes)"
            for p in entries
        )

    def read_file(self, path: str, offset: int = 0, limit: int = 400) -> str:
        """Read a text file from your workspace.

        Args:
            path: File path relative to the workspace root.
            offset: Line number to start from (0-based).
            limit: Maximum number of lines to return.
        """
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"{path} does not exist")
        lines = p.read_text(errors="replace").splitlines()
        chunk = "\n".join(lines[offset:offset + limit])
        if offset + limit < len(lines):
            chunk += f"\n[... {len(lines) - offset - limit} more lines; use offset to read further]"
        return chunk or "(empty file)"

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a file in your workspace.

        Args:
            path: File path relative to the workspace root. Parent directories are created.
            content: The full file content.
        """
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {self._rel(p)}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace one exact occurrence of old_text with new_text in a workspace file.

        Args:
            path: File path relative to the workspace root.
            old_text: Exact text to find. Must occur exactly once in the file.
            new_text: Replacement text.
        """
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"{path} does not exist")
        s = p.read_text()
        n = s.count(old_text)
        if n != 1:
            raise ToolError(f"old_text occurs {n} times in {path}; it must occur exactly once")
        p.write_text(s.replace(old_text, new_text))
        return f"Edited {self._rel(p)}"

    def delete_file(self, path: str) -> str:
        """Delete a file or an empty directory in your workspace.

        Args:
            path: Path relative to the workspace root.
        """
        p = self._path(path)
        if p == self.root:
            raise ToolError("cannot delete the workspace root")
        if p.is_dir():
            p.rmdir()
        elif p.exists():
            p.unlink()
        else:
            raise ToolError(f"{path} does not exist")
        return f"Deleted {self._rel(p)}"

    # Web

    def web_search(self, query: str) -> str:
        """Search the web.

        Args:
            query: The search query.
        """
        from ddgs import DDGS

        results = DDGS().text(query, max_results=self.cfg.search_results)
        if not results:
            return "No results."
        return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

    def fetch_url(self, url: str) -> str:
        """Fetch a web page and return its main text content.

        Args:
            url: The http(s) URL to fetch.
        """
        with httpx.Client(timeout=self.cfg.fetch_timeout, follow_redirects=False,
                          headers={"User-Agent": "Mozilla/5.0 (entity research agent)"}) as client:
            for _ in range(6):
                _check_public(url)
                r = client.get(url)
                if r.is_redirect and "location" in r.headers:
                    url = urljoin(url, r.headers["location"])
                    continue
                break
            else:
                raise ToolError("too many redirects")
        ctype = r.headers.get("content-type", "")
        if r.status_code >= 400:
            raise ToolError(f"HTTP {r.status_code}")
        if "html" in ctype:
            text = trafilatura.extract(r.text, include_links=True, url=url) or r.text
        elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
            text = r.text
        else:
            raise ToolError(f"unsupported content type: {ctype or 'unknown'}")
        if len(text) > self.cfg.fetch_max_chars:
            text = text[:self.cfg.fetch_max_chars] + f"\n[... truncated, {len(text)} chars total]"
        return f"{url}\n\n{text}"

    # Memory

    def recall(self, query: str, k: int = 5) -> str:
        """Search your long-term memories.

        Args:
            query: What you are trying to remember.
            k: How many memories to return.
        """
        found = self.memory.search(query, min(int(k), 20))
        return "\n".join(m.render() for m in found) or "Nothing comes to mind."


def _check_public(url: str):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError("only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ToolError(f"could not resolve {u.hostname}: {e}")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ToolError(f"{u.hostname} resolves to a non-public address")


def _schema(f) -> dict:
    """Ollama's converter marks every parameter as required; fix that using the signature."""
    schema = convert_function_to_tool(f).model_dump(exclude_none=True)
    params = inspect.signature(f).parameters
    schema["function"]["parameters"]["required"] = [
        n for n, p in params.items() if p.default is inspect.Parameter.empty
    ]
    return schema
