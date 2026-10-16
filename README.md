# entity

A continuously running local LLM agent for experimenting with loops and memory.

There's no user and no prompts — `entity` runs on its own heartbeat, driven by a local
[Ollama](https://ollama.com) model. Each tick it decides for itself whether to act: read or
write files in its workspace, search and fetch pages from the web, or just wait. Older
context fades and gets cut (`rot`) as the conversation grows, and anything worth keeping is
extracted into long-term memory that it can later `recall`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally, with the models from `config.toml` pulled:

```bash
ollama pull qwen3.6:35b-a3b
ollama pull qwen3-embedding:0.6b
```

## Running

```bash
uv sync
uv run python -m entity
```

By default it loads `config.toml` from the current directory. Pass `-c/--config` to use a
different file:

```bash
uv run python -m entity -c path/to/config.toml
```

The agent's workspace, memory, and run logs live under `workspace/`, `data/`, and `runs/`
respectively (created relative to the config file).
