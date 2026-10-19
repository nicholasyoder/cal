"""Run logging: every event goes to runs/<id>/events.jsonl, with a readable echo to the console."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

DIM, CYAN, YELLOW, GREEN, MAGENTA, RED, BLUE, RESET = (
    "\033[2m", "\033[36m", "\033[33m", "\033[32m", "\033[35m", "\033[31m", "\033[34m", "\033[0m",
)


class RunLog:
    def __init__(self, runs_dir: Path):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = (self.dir / "events.jsonl").open("a", buffering=1)

    def event(self, type: str, **data):
        data = {"t": time.time(), "type": type, **data}
        self._events.write(json.dumps(data, default=str, ensure_ascii=False) + "\n")

    def snapshot(self, messages: list[dict]):
        """Overwrite context.json with what the model currently sees."""
        tmp = self.dir / "context.json.tmp"
        tmp.write_text(json.dumps(messages, indent=1, ensure_ascii=False, default=str))
        tmp.replace(self.dir / "context.json")

    # Console helpers

    def stream(self, text: str, color: str = ""):
        sys.stdout.write(f"{color}{text}{RESET if color else ''}")
        sys.stdout.flush()

    def line(self, text: str, color: str = ""):
        self.stream(text + "\n", color)
