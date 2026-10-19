# CAL (Continuous Agentic Loop)

A continuously running local LLM agent for experimenting with loops and memory.

There's no user and no prompts — `CAL` runs on its own heartbeat, driven by a local
[Ollama](https://ollama.com) model (or, with a config change, any OpenAI-compatible hosted
endpoint — see below). Each tick it decides for itself whether to act: read or write files
in its workspace, search and fetch pages from the web, or just wait. Older context fades and
gets cut (`rot`) as the conversation grows, and anything worth keeping is extracted into
long-term memory that it can later `recall`.

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
uv run python -m cal
```

By default it loads `config.toml` from the current directory. Pass `-c/--config` to use a
different file:

```bash
uv run python -m cal -c path/to/config.toml
```

The agent's workspace, memory, and run logs live under `workspace/`, `data/`, and `runs/`
respectively (created relative to the config file).

## Using a hosted model instead of Ollama

Chat (though not embeddings — those stay on a local Ollama server, see `[memory]` in
`config.toml`) can be offloaded to any OpenAI-compatible endpoint, such as
[NanoGPT](https://docs.nano-gpt.com/introduction), OpenRouter, or a self-hosted vLLM server.
In `config.toml`:

```toml
[model]
provider = "openai"
name = "deepseek-v3.2"                 # whatever model name the endpoint expects
host = "https://nano-gpt.com/api/v1"   # the API base URL
api_key_env = "NANOGPT_API_KEY"        # name of the env var holding the key
```

Export the key before running (`export NANOGPT_API_KEY=...`) rather than putting it in the
config file. See `cal/config.py`'s `ModelCfg` for the full set of options.
