"""Long-term memory: an embedded JSONL store, plus an extractor that mines the transcript."""

import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ollama

from .backend import Backend
from .config import Config

QUERY_INSTRUCT = "Instruct: Given an agent's current situation, retrieve its relevant memories\nQuery: "


@dataclass
class Memory:
    id: str
    text: str
    importance: int
    created: float
    run: str
    step: int

    def render(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created))
        return f"[{when}] {self.text}"


class MemoryStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg.memory
        # Embeddings always go to a local Ollama server, independent of `model.provider`.
        self.client = ollama.Client(host=cfg.memory.embed_host)
        self.path = cfg.data_dir / "memories.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: list[Memory] = []
        vecs = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    vecs.append(d.pop("embedding"))
                    self.items.append(Memory(**d))
        self.vecs = np.array(vecs, dtype=np.float32) if vecs else np.empty((0, 0), np.float32)

    def __len__(self):
        return len(self.items)

    def embed(self, texts: list[str]) -> np.ndarray:
        opts = {"num_gpu": 0} if self.cfg.embed_on_cpu else None
        r = self.client.embed(model=self.cfg.embed_model, input=texts, options=opts)
        v = np.array(r.embeddings, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def add(self, text: str, importance: int, run: str, step: int) -> Memory | None:
        """Store a memory unless a near-duplicate already exists."""
        v = self.embed([text])[0]
        if len(self.items) and float((self.vecs @ v).max()) >= self.cfg.dedupe_threshold:
            return None
        m = Memory(uuid.uuid4().hex[:12], text, importance, time.time(), run, step)
        self.items.append(m)
        self.vecs = np.vstack([self.vecs, v[None]]) if len(self.vecs) else v[None]
        with self.path.open("a") as f:
            f.write(json.dumps({**m.__dict__, "embedding": v.round(6).tolist()}) + "\n")
        return m

    def search(self, query: str, k: int) -> list[Memory]:
        if not self.items or not query.strip() or k <= 0:
            return []
        q = self.embed([QUERY_INSTRUCT + query[-4000:]])[0]
        scores = self.vecs @ q
        return [self.items[i] for i in np.argsort(-scores)[:k]]

    def recent(self, k: int) -> list[Memory]:
        return self.items[-k:] if k > 0 else []

    def sample(self, k: int) -> list[Memory]:
        """A random, relevance-free handful of memories — dream material."""
        if not self.items or k <= 0:
            return []
        return random.sample(self.items, min(k, len(self.items)))

    def surface(self, query: str) -> list[Memory]:
        """Memories to inject into a heartbeat: relevant ones plus the latest few."""
        seen, out = set(), []
        for m in self.search(query, self.cfg.inject_relevant) + self.recent(self.cfg.inject_recent):
            if m.id not in seen:
                seen.add(m.id)
                out.append(m)
        return sorted(out, key=lambda m: m.created)


EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["text", "importance"],
            },
        }
    },
    "required": ["memories"],
}


class Extractor:
    def __init__(self, cfg: Config, backend: Backend, store: MemoryStore, prompt: str):
        self.cfg = cfg
        self.backend = backend
        self.store = store
        self.prompt = prompt

    def extract(self, transcript: str, run: str, step: int) -> tuple[list[Memory], list[str]]:
        """Returns (stored memories, candidates skipped as duplicates)."""
        # Leave room for the prompt and output within the context window.
        budget = int(self.cfg.model.num_ctx * self.cfg.rot.chars_per_token * 0.6)
        transcript = transcript[-budget:]
        existing = self.store.search(transcript, 12)
        existing_txt = "\n".join(f"- {m.text}" for m in existing) or "(none)"
        r = self.backend.chat(
            model=self.cfg.memory.extract_model or self.cfg.model.name,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"## Existing related memories\n{existing_txt}\n\n## Transcript segment\n{transcript}"},
            ],
            format=EXTRACT_SCHEMA,
            think=self.cfg.memory.extract_think,
            num_ctx=self.cfg.model.num_ctx,
            options={**self.cfg.model.options, "temperature": 0.2},
        )
        try:
            items = json.loads(r.content)["memories"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [], []
        stored, skipped = [], []
        for it in items:
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            m = self.store.add(text, int(it.get("importance", 3)), run, step)
            (stored if m else skipped).append(m or text)
        return stored, skipped
