"""Model backends: native Ollama, or any OpenAI-compatible endpoint (NanoGPT, OpenRouter,
vLLM, etc). Both normalize to the same `ChatChunk`/`ChatResponse` shapes so the rest of the
code — `loop.py`, `memory.py` — doesn't know or care which one it's talking to.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np
import ollama

from .config import ModelCfg


class BackendError(Exception):
    """Wraps any provider-specific request failure, so callers can catch one type."""


@dataclass
class ChatChunk:
    thinking: str = ""
    content: str = ""
    # Populated (non-empty) only once a tool call is fully known; empty on every other chunk.
    tool_calls: list[dict] = field(default_factory=list)
    done: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    duration_s: float | None = None


@dataclass
class ChatResponse:
    thinking: str = ""
    content: str = ""


class Backend(Protocol):
    def chat_stream(self, *, model: str, messages: list[dict], tools: list[dict],
                     think: bool, num_ctx: int, options: dict) -> Iterator[ChatChunk]: ...

    def chat(self, *, model: str, messages: list[dict], think: bool, num_ctx: int,
              options: dict, format: Any = None) -> ChatResponse: ...

    def embed(self, *, model: str, texts: list[str], on_cpu: bool) -> np.ndarray: ...


def make_backend(cfg: ModelCfg) -> Backend:
    if cfg.provider == "ollama":
        return OllamaBackend(cfg)
    if cfg.provider == "openai":
        return OpenAIBackend(cfg)
    raise ValueError(f"unknown model provider: {cfg.provider!r} (want 'ollama' or 'openai')")


# --- Ollama -----------------------------------------------------------------------------

class OllamaBackend:
    """Talks to a native Ollama server (the default)."""

    def __init__(self, cfg: ModelCfg):
        self.client = ollama.Client(host=cfg.host)

    def chat_stream(self, *, model, messages, tools, think, num_ctx, options):
        try:
            stream = self.client.chat(
                model=model, messages=messages, tools=tools, think=think, stream=True,
                options={**options, "num_ctx": num_ctx},
            )
            for chunk in stream:
                m = chunk.message
                yield ChatChunk(
                    thinking=m.thinking or "",
                    content=m.content or "",
                    tool_calls=[
                        {"id": uuid.uuid4().hex, "function": {"name": tc.function.name,
                                                               "arguments": dict(tc.function.arguments or {})}}
                        for tc in (m.tool_calls or [])
                    ],
                    done=bool(chunk.done),
                    prompt_tokens=chunk.prompt_eval_count if chunk.done else None,
                    output_tokens=chunk.eval_count if chunk.done else None,
                    duration_s=(chunk.total_duration or 0) / 1e9 if chunk.done else None,
                )
        except (ollama.ResponseError, ConnectionError) as e:
            raise BackendError(str(e)) from e

    def chat(self, *, model, messages, think, num_ctx, options, format=None):
        try:
            r = self.client.chat(
                model=model, messages=messages, think=think, format=format,
                options={**options, "num_ctx": num_ctx},
            )
        except (ollama.ResponseError, ConnectionError) as e:
            raise BackendError(str(e)) from e
        return ChatResponse(thinking=r.message.thinking or "", content=r.message.content or "")

    def embed(self, *, model, texts, on_cpu):
        try:
            r = self.client.embed(model=model, input=texts, options={"num_gpu": 0} if on_cpu else None)
        except (ollama.ResponseError, ConnectionError) as e:
            raise BackendError(str(e)) from e
        return np.array(r.embeddings, dtype=np.float32)


# --- OpenAI-compatible --------------------------------------------------------------------

class OpenAIBackend:
    """Talks to any OpenAI-compatible chat-completions endpoint: NanoGPT, OpenRouter, vLLM,
    LM Studio, etc. `cfg.host` is the API base URL (e.g. "https://nano-gpt.com/api/v1") and
    `cfg.api_key_env` names the environment variable holding the API key.

    `num_ctx` is accepted but ignored — a hosted model's context window is fixed server-side;
    it's still used locally to decide when `rot` should start cutting.
    """

    def __init__(self, cfg: ModelCfg):
        from openai import OpenAI  # imported lazily so the dependency is optional

        api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else None
        if cfg.api_key_env and not api_key:
            raise BackendError(f"${cfg.api_key_env} is not set (required for provider 'openai')")
        self.client = OpenAI(base_url=cfg.host, api_key=api_key or "unused")
        self.reasoning_effort = cfg.reasoning_effort

    def _extra_body(self) -> dict:
        return {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}

    def chat_stream(self, *, model, messages, tools, think, num_ctx, options):
        from openai import APIError, APIConnectionError

        msgs = _to_openai_messages(messages)
        tool_defs = [_to_openai_tool(t) for t in tools] if tools else None
        kwargs = dict(model=model, messages=msgs, stream=True,
                      stream_options={"include_usage": True})
        if tool_defs:
            kwargs["tools"] = tool_defs
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if think and self.reasoning_effort:
            kwargs["extra_body"] = self._extra_body()
        try:
            stream = self.client.chat.completions.create(**kwargs)
            acc: dict[int, dict] = {}
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if not chunk.choices:
                    if usage:
                        yield ChatChunk(done=True, prompt_tokens=usage.prompt_tokens,
                                        output_tokens=usage.completion_tokens)
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                thinking = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
                content = delta.content or ""
                for tc in (delta.tool_calls or []):
                    slot = acc.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
                done = choice.finish_reason is not None
                yield ChatChunk(thinking=thinking, content=content, done=False)
                if done:
                    calls = []
                    for idx in sorted(acc):
                        slot = acc[idx]
                        try:
                            args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        calls.append({"id": slot["id"] or uuid.uuid4().hex,
                                      "function": {"name": slot["name"], "arguments": args}})
                    yield ChatChunk(
                        tool_calls=calls, done=True,
                        prompt_tokens=usage.prompt_tokens if usage else None,
                        output_tokens=usage.completion_tokens if usage else None,
                    )
        except (APIError, APIConnectionError) as e:
            raise BackendError(str(e)) from e

    def chat(self, *, model, messages, think, num_ctx, options, format=None):
        from openai import APIError, APIConnectionError

        msgs = _to_openai_messages(messages)
        kwargs = dict(model=model, messages=msgs)
        if "temperature" in options:
            kwargs["temperature"] = options["temperature"]
        if "top_p" in options:
            kwargs["top_p"] = options["top_p"]
        if think and self.reasoning_effort:
            kwargs["extra_body"] = self._extra_body()
        if format:
            # Best-effort: most OpenAI-compatible providers accept a JSON-schema response
            # format this way; providers that don't will just ignore/reject it. `strict`
            # is left off rather than set true — our schema doesn't set
            # `additionalProperties: false` throughout, which strict mode requires, and
            # many third-party providers reject or mishandle it anyway.
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "extract", "schema": format}}
        try:
            r = self.client.chat.completions.create(**kwargs)
        except (APIError, APIConnectionError) as e:
            raise BackendError(str(e)) from e
        msg = r.choices[0].message
        thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        return ChatResponse(thinking=thinking, content=msg.content or "")

    def embed(self, *, model, texts, on_cpu):
        from openai import APIError, APIConnectionError

        try:
            r = self.client.embeddings.create(model=model, input=texts)
        except (APIError, APIConnectionError) as e:
            raise BackendError(str(e)) from e
        return np.array([d.embedding for d in r.data], dtype=np.float32)


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {"id": c.get("id") or uuid.uuid4().hex, "type": "function",
                     "function": {"name": c["function"]["name"],
                                  "arguments": json.dumps(c["function"]["arguments"], ensure_ascii=False)}}
                    for c in m["tool_calls"]
                ],
            })
        elif role == "tool":
            out.append({
                "role": "tool",
                "content": m.get("content", ""),
                "tool_call_id": m.get("tool_call_id") or uuid.uuid4().hex,
                "name": m.get("tool_name", ""),
            })
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _to_openai_tool(t: dict) -> dict:
    # Our schemas are already built in OpenAI's function-tool shape (see tools.py); pass through.
    return t
