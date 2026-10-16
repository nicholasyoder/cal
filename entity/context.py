"""The working context the model sees, and the rot that eats it.

Once the estimated size passes `trigger`, random spans are replaced with "…" until it is
back under `target`. Cuts are weighted towards the oldest messages, so old text turns to
swiss cheese and then disappears, while recent text is mostly intact.

Messages that are fully destroyed (not just cut to swiss cheese) leave no trace of their
content, but their loss is tallied and reported back as a standing notice prepended to
`messages` — so the model has some sense of how much history is actually gone, even where
it can't recall what. The notice is popped off before each rot pass (so it's never itself a
cutting/drop candidate) and rebuilt fresh afterwards.
"""

import json
import math
import random
import re
import time

from .config import RotCfg

ELLIPSIS = "…"
_RUNS = re.compile(r"…(?:\s*…)+")


class Context:
    def __init__(self, system: str, cfg: RotCfg, num_ctx: int, tool_schema_chars: int = 0):
        self.system = system
        self.cfg = cfg
        self.num_ctx = num_ctx
        self.messages: list[dict] = []
        self.timestamps: list[float] = []
        self.rng = random.Random(cfg.seed)
        self._overhead = tool_schema_chars
        self._origin_ts: float | None = None
        self._destroyed = 0
        self._destroyed_chars = 0
        self._destroyed_span_end: float | None = None
        self._has_notice = False

    def add(self, msg: dict):
        self.messages.append(msg)
        now = time.time()
        self.timestamps.append(now)
        if self._origin_ts is None:
            self._origin_ts = now

    def render(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, *self.messages]

    def est_tokens(self) -> int:
        chars = len(self.system) + self._overhead
        for m in self.messages:
            chars += len(m.get("content", "")) + 16
            if m.get("tool_calls"):
                chars += len(json.dumps(m["tool_calls"]))
        return int(chars / self.cfg.chars_per_token)

    def rot(self) -> dict | None:
        cfg = self.cfg
        before = self.est_tokens()
        if before < cfg.trigger * self.num_ctx:
            return None

        if self._has_notice:
            self.messages.pop(0)
            self.timestamps.pop(0)
            self._has_notice = False

        to_remove = (before - cfg.target * self.num_ctx) * cfg.chars_per_token
        removed = cuts = dropped = 0
        while removed < to_remove:
            n = len(self.messages) - cfg.protect_recent
            if n <= 0:
                break
            i = self._pick(n)
            if i is None:
                break
            m = _flatten(self.messages[i])
            self.messages[i] = m
            old = m["content"]
            new = self._cut(old)
            if _surviving(new) < cfg.min_keep:
                del self.messages[i]
                self._destroy(self.timestamps.pop(i), len(old))
                removed += len(old)
                dropped += 1
            else:
                m["content"] = new
                removed += len(old) - len(new)
            cuts += 1

        # Safety net: never let the protected tail push us past the hard limit.
        while self.est_tokens() > cfg.hard * self.num_ctx and len(self.messages) > 1:
            self._destroy(self.timestamps.pop(0), len(self.messages[0].get("content", "")))
            del self.messages[0]
            dropped += 1

        if cfg.notice and self._destroyed:
            self.messages.insert(0, {"role": "user", "content": self._notice_text()})
            self.timestamps.insert(0, self._destroyed_span_end)
            self._has_notice = True

        return {"tokens_before": before, "tokens_after": self.est_tokens(), "cuts": cuts,
                "dropped": dropped, "messages": len(self.messages),
                "destroyed_total": self._destroyed, "destroyed_span_s": self._span_seconds()}

    def _destroy(self, ts: float, chars: int):
        self._destroyed += 1
        self._destroyed_chars += chars
        self._destroyed_span_end = max(self._destroyed_span_end or ts, ts)

    def _span_seconds(self) -> float:
        if self._destroyed_span_end is None or self._origin_ts is None:
            return 0.0
        return self._destroyed_span_end - self._origin_ts

    def _notice_text(self) -> str:
        tokens = int(self._destroyed_chars / self.cfg.chars_per_token)
        return (f"[{_fmt_span(self._span_seconds())} of earlier session log destroyed by "
                f"memory rot — {self._destroyed} messages, ~{tokens:,} tokens gone for good]")

    def _pick(self, n: int) -> int | None:
        """Choose a message index in [0, n), favouring older and longer messages."""
        weights = [
            math.exp(-self.cfg.bias * i / n) * max(len(self.messages[i].get("content", "")), 1)
            for i in range(n)
        ]
        if not any(weights):
            return None
        return self.rng.choices(range(n), weights=weights)[0]

    def _cut(self, s: str) -> str:
        if not s:
            return s
        length = self.rng.randint(self.cfg.cut_min, self.cfg.cut_max)
        start = self.rng.randrange(len(s))
        return _RUNS.sub(ELLIPSIS, s[:start] + ELLIPSIS + s[start + length:])


def _surviving(s: str) -> int:
    return len(s.replace(ELLIPSIS, "").strip())


def _fmt_span(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m"
    return f"{s}s"


def _flatten(m: dict) -> dict:
    """Fold structured tool calls into plain text so they can be cut like anything else."""
    if not m.get("tool_calls"):
        return m
    calls = "\n".join(
        f"[called {c['function']['name']}({json.dumps(c['function']['arguments'], ensure_ascii=False)})]"
        for c in m["tool_calls"]
    )
    content = f"{m.get('content', '')}\n{calls}".strip()
    return {"role": m["role"], "content": content}
