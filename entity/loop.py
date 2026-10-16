"""The main loop: heartbeat → model → tools → … → turn ends → extract memories → heartbeat.

Two histories are kept: `ctx` is what the model sees, and it rots; `pending` is the full,
uncut record since the last memory extraction, so extraction always sees the real thing.
"""

import json
import time
from datetime import datetime

import httpx
import ollama

from .config import Config
from .context import Context
from .log import CYAN, DIM, GREEN, MAGENTA, RED, YELLOW, RunLog
from .memory import Extractor, MemoryStore
from .tools import Toolbox

EXTRACT_RESULT_CHARS = 1500


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = ollama.Client(host=cfg.model.host)
        self.log = RunLog(cfg.runs_dir)
        self.memory = MemoryStore(cfg, self.client)
        self.tools = Toolbox(cfg, self.memory)
        prompts = cfg.prompts_dir
        self.heartbeat_tpl = (prompts / "heartbeat.md").read_text()
        self.dream_tpl = (prompts / "dream.md").read_text()
        self.extractor = Extractor(cfg, self.client, self.memory, (prompts / "extract.md").read_text())
        self.ctx = Context(
            (prompts / "system.md").read_text(), cfg.rot, cfg.model.num_ctx,
            tool_schema_chars=len(json.dumps(self.tools.schemas)),
        )
        self.pending: list[str] = []
        self.step = 0
        self.turn = 0
        self.steps_since_extract = 0
        self.last_output = ""

    def run(self):
        self.log.event("run_start", run=self.log.run_id, config=self.cfg, memories=len(self.memory))
        self.log.line(f"run {self.log.run_id} · {len(self.memory)} memories · logs in {self.log.dir}", DIM)
        self._advance_turn()
        try:
            while True:
                self._step()
        except KeyboardInterrupt:
            self.log.line("\ninterrupted — extracting pending memories before exit (Ctrl+C again to skip)", YELLOW)
            try:
                self._extract()
            except KeyboardInterrupt:
                pass
            self.log.event("run_end", step=self.step)

    def _advance_turn(self):
        """Move to the next turn: a dream every `every_turns` turns, a heartbeat otherwise.

        A dream turn is consumed entirely by the dream — no heartbeat, model step, or
        extraction happens "on" it — so this loops straight through to the next turn, which
        is always a normal heartbeat whose context already contains the labeled dream.
        """
        while True:
            self.turn += 1
            if self.cfg.dream.every_turns and self.turn % self.cfg.dream.every_turns == 0:
                self._dream()
                continue
            self._heartbeat()
            return

    def _dream(self):
        mems = self.memory.sample(self.cfg.dream.memory_count)
        mem_txt = "\n".join(f"- {m.render()}" for m in mems) or "(nothing surfaces)"
        try:
            r = self.client.chat(
                model=self.cfg.model.name,
                messages=[
                    {"role": "system", "content": self.dream_tpl},
                    {"role": "user", "content": f"Fragments surfacing:\n{mem_txt}"},
                ],
                think=self.cfg.dream.think,
                options={**self.cfg.model.options, "num_ctx": self.cfg.model.num_ctx,
                        "temperature": self.cfg.dream.temperature},
            )
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as e:
            self.log.event("error", step=self.step, error=f"dream: {e}")
            self.log.line(f"\n[dream failed: {e}]", RED)
            return
        text = (r.message.content or "").strip()
        if not text:
            return
        labeled = f"[dream — turn {self.turn}]\n{text}\n[/dream]"
        self.ctx.add({"role": "user", "content": labeled})
        self.log.event("dream", turn=self.turn, step=self.step, memories=[m.id for m in mems],
                       thinking=r.message.thinking, content=text)
        self.log.line(f"\n~~ dream · turn {self.turn} ~~\n{text}\n", MAGENTA)

    def _heartbeat(self):
        surfaced = self.memory.surface(self.last_output)
        mem_txt = "\n".join(f"- {m.render()}" for m in surfaced) or "(nothing)"
        text = (self.heartbeat_tpl
                .replace("{time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                .replace("{turn}", str(self.turn))
                .replace("{memories}", mem_txt)).strip()
        self.ctx.add({"role": "user", "content": text})
        self.pending.append(f"[prompt]\n{text}")
        self.log.event("heartbeat", turn=self.turn, step=self.step, content=text,
                       surfaced=[m.id for m in surfaced])
        self.log.line(f"\n━━ turn {self.turn} ━━ {len(surfaced)} memories surfaced", CYAN)

    def _step(self):
        self.step += 1
        try:
            thinking, content, calls, stats = self._chat()
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as e:
            self.log.event("error", step=self.step, error=str(e))
            self.log.line(f"\n[model error: {e}] retrying", RED)
            time.sleep(3)
            return

        # Ollama's renderer drops the `thinking` field once a turn is done, so to keep the
        # model's reasoning in its history it has to live in the message text instead —
        # where it also becomes subject to rot and recall like everything else.
        visible = content
        if self.cfg.loop.keep_thinking and thinking.strip():
            visible = f"<reasoning>\n{thinking.strip()}\n</reasoning>\n{content}".strip()
        msg = {"role": "assistant", "content": visible}
        if calls:
            msg["tool_calls"] = calls
        self.ctx.add(msg)
        self.log.event("response", step=self.step, thinking=thinking, content=content,
                       tool_calls=calls, **stats)
        if thinking:
            self.pending.append(f"[thinking]\n{thinking}")
        if content:
            self.pending.append(f"[response]\n{content}")
        self.last_output = f"{thinking[-2000:]}\n{content}"

        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            self.log.line(f"\n→ {name}({_short(json.dumps(args, ensure_ascii=False), 300)})", YELLOW)
            result = self.tools.call(name, args)
            if len(result) > self.cfg.loop.tool_result_max_chars:
                result = result[:self.cfg.loop.tool_result_max_chars] + "\n[... truncated]"
            self.log.line(_short(result, 500), DIM)
            self.ctx.add({"role": "tool", "content": result, "tool_name": name})
            self.log.event("tool", step=self.step, name=name, args=args, result=result)
            self.pending.append(f"[tool call] {name}({json.dumps(args, ensure_ascii=False)})\n"
                                f"[result]\n{_short(result, EXTRACT_RESULT_CHARS)}")

        self.steps_since_extract += 1
        if not calls:
            self._extract()
            if self.cfg.loop.turn_delay:
                time.sleep(self.cfg.loop.turn_delay)
            self._advance_turn()
        elif self.steps_since_extract >= self.cfg.loop.extract_every_steps:
            self._extract()

        rot = self.ctx.rot()
        if rot:
            self.log.event("rot", step=self.step, **rot)
            self.log.line(f"\n[rot: {rot['tokens_before']}→{rot['tokens_after']} tokens, "
                          f"{rot['cuts']} cuts, {rot['dropped']} messages gone — "
                          f"{rot['destroyed_total']} destroyed total, "
                          f"{rot['destroyed_span_s'] / 60:.0f}m span]", MAGENTA)
        self.log.snapshot(self.ctx.render())

    def _chat(self):
        thinking, content, calls, final = [], [], [], None
        mode = None
        stream = self.client.chat(
            model=self.cfg.model.name,
            messages=self.ctx.render(),
            tools=self.tools.schemas,
            think=self.cfg.model.think,
            stream=True,
            options={**self.cfg.model.options, "num_ctx": self.cfg.model.num_ctx},
        )
        for chunk in stream:
            m = chunk.message
            if m.thinking:
                if mode != "think":
                    self.log.stream("\n", DIM)
                    mode = "think"
                thinking.append(m.thinking)
                self.log.stream(m.thinking, DIM)
            if m.content:
                if mode != "say":
                    self.log.stream("\n")
                    mode = "say"
                content.append(m.content)
                self.log.stream(m.content)
            for tc in m.tool_calls or []:
                calls.append({"function": {"name": tc.function.name,
                                           "arguments": dict(tc.function.arguments or {})}})
            if chunk.done:
                final = chunk
        stats = {}
        if final is not None:
            stats = {"prompt_tokens": final.prompt_eval_count, "output_tokens": final.eval_count,
                     "duration_s": (final.total_duration or 0) / 1e9,
                     "est_context_tokens": self.ctx.est_tokens()}
        return "".join(thinking), "".join(content), calls, stats

    def _extract(self):
        self.steps_since_extract = 0
        if not self.pending:
            return
        transcript = "\n\n".join(self.pending)
        self.pending = []
        self.log.line("\n[extracting memories…]", DIM)
        try:
            stored, skipped = self.extractor.extract(transcript, self.log.run_id, self.step)
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as e:
            self.log.event("error", step=self.step, error=f"extraction: {e}")
            self.log.line(f"[extraction failed: {e}]", RED)
            return
        self.log.event("extract", step=self.step, transcript_chars=len(transcript),
                       stored=[m.__dict__ for m in stored], skipped=skipped)
        for m in stored:
            self.log.line(f"  + {m.text}", GREEN)
        if skipped:
            self.log.line(f"  ({len(skipped)} duplicates skipped)", DIM)


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f" …[{len(s) - n} more]"
