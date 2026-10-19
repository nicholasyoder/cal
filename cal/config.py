"""Configuration: dataclass defaults, overridden by sections in config.toml."""

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


@dataclass
class ModelCfg:
    name: str = "gpt-oss:latest"
    # "ollama" talks to a native Ollama server; "openai" talks to any OpenAI-compatible
    # chat-completions endpoint (NanoGPT, OpenRouter, vLLM, LM Studio, ...).
    provider: str = "ollama"
    # Ollama: the server host. OpenAI-compatible: the API base URL, e.g.
    # "https://nano-gpt.com/api/v1".
    host: str = "http://localhost:11434"
    # OpenAI-compatible only: name of the environment variable holding the API key.
    # Leave empty for a local/keyless endpoint.
    api_key_env: str = ""
    num_ctx: int = 16384
    think: bool = True
    # OpenAI-compatible only: passed as `reasoning_effort` when `think` is true, for
    # providers/models that gate reasoning behind it (e.g. "low"/"medium"/"high"). Leave
    # empty to send nothing and let the model's own default apply.
    reasoning_effort: str = ""
    # Passed straight through to Ollama's `options`; for an OpenAI-compatible provider only
    # `temperature` and `top_p` are used (everything else is Ollama-specific and ignored).
    options: dict = field(default_factory=lambda: {"temperature": 0.7, "top_p": 0.95})


@dataclass
class LoopCfg:
    # Seconds to wait between turns (a turn ends when the model replies without calling a tool).
    turn_delay: float = 0.0
    # Run memory extraction mid-turn after this many consecutive tool-calling steps.
    extract_every_steps: int = 8
    # Hard cap on characters of a single tool result placed into context.
    tool_result_max_chars: int = 12000
    # Fold each turn's reasoning back into the visible message text, so it survives into
    # history (Ollama otherwise drops `thinking` once a turn is done) and gets cut/recalled
    # like everything else instead of vanishing the instant the turn ends.
    keep_thinking: bool = True
    # Workspace file (edited by the agent with its normal file tools) whose contents are
    # always appended to the system prompt, immune to rot — a standing note to itself.
    soul_file: str = "SOUL.md"


@dataclass
class RotCfg:
    # Fractions of num_ctx. Rot begins once the estimate passes `trigger`, then cuts until
    # the context is back down to `target`. The gap between the two keeps cuts in bursts,
    # so Ollama's prompt cache stays valid between them.
    trigger: float = 0.6
    target: float = 0.45
    # Above this, whole messages are dropped oldest-first regardless of randomness.
    hard: float = 0.85
    # The most recent N messages are never cut.
    protect_recent: int = 4
    # How strongly cuts favour older messages (0 = uniform; higher = more front-loaded).
    bias: float = 4.0
    cut_min: int = 8
    cut_max: int = 160
    # A message with fewer surviving characters than this is removed entirely.
    min_keep: int = 24
    # Rough chars-per-token used to estimate context size.
    chars_per_token: float = 3.2
    # Prepend a standing notice tallying how much has been fully destroyed (not just
    # cut to swiss cheese), so the model has a sense of scale even where it can't recall.
    notice: bool = True
    seed: int | None = None


@dataclass
class MemoryCfg:
    # Embeddings always run on a local Ollama server, regardless of `model.provider` — this
    # keeps the vector store self-consistent even when chat is offloaded to a hosted model.
    embed_host: str = "http://localhost:11434"
    embed_model: str = "qwen3-embedding:0.6b"
    # Run the embedder on CPU so it doesn't evict the main model from VRAM.
    embed_on_cpu: bool = True
    # Model used for extraction; empty means the main model. Runs on `model.provider`.
    extract_model: str = ""
    extract_think: bool = False
    # Memories surfaced in each heartbeat: top-k by relevance plus the most recent few.
    inject_relevant: int = 6
    inject_recent: int = 3
    # New memories more similar than this to an existing one are skipped.
    dedupe_threshold: float = 0.85


@dataclass
class DreamCfg:
    # Turns between dreams; 0 disables. Must be >= 2 — 1 would mean every turn is a dream and
    # the loop never reaches a normal heartbeat.
    every_turns: int = 5
    # Random memories sampled as dream material (no relevance filter, unlike heartbeat surfacing).
    memory_count: int = 8
    think: bool = False
    temperature: float = 1.0


@dataclass
class SubconsciousCfg:
    enabled: bool = True
    # Empty means the main model; can be pointed at something else (e.g. a cheaper model,
    # like `memory.extract_model`) independently of it.
    model: str = ""
    think: bool = False
    temperature: float = 0.9
    # A couple of random memories tossed in alongside the transcript, dream-style, so it has
    # something unrelated to redirect toward if the agent's gone narrow.
    memory_count: int = 2


@dataclass
class ToolsCfg:
    fetch_max_chars: int = 10000
    fetch_timeout: float = 20.0
    search_results: int = 6


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    loop: LoopCfg = field(default_factory=LoopCfg)
    rot: RotCfg = field(default_factory=RotCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    dream: DreamCfg = field(default_factory=DreamCfg)
    subconscious: SubconsciousCfg = field(default_factory=SubconsciousCfg)
    tools: ToolsCfg = field(default_factory=ToolsCfg)
    root: Path = Path(".")
    # Where workspace/data/runs live for this run; defaults to `root` but can point
    # elsewhere so the same config and prompts can be reused across separate runs.
    base: Path | None = None

    @property
    def workspace(self) -> Path:
        return (self.base or self.root) / "workspace"

    @property
    def data_dir(self) -> Path:
        return (self.base or self.root) / "data"

    @property
    def runs_dir(self) -> Path:
        return (self.base or self.root) / "runs"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"


def load_config(path: Path, base: Path | None = None) -> Config:
    cfg = Config(root=path.parent.resolve(), base=base.resolve() if base else None)
    if not path.exists():
        return cfg
    raw = tomllib.loads(path.read_text())
    for f in fields(cfg):
        if f.name in raw:
            setattr(cfg, f.name, replace(getattr(cfg, f.name), **raw[f.name]))
    return cfg
