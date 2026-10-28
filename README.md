# Dummy OpenAI API

A fake OpenAI-compatible HTTP server that returns scripted responses. Point your app's OpenAI client at it during development and tests, so you don't spend tokens, need network access, or get non-deterministic output.

## Supported endpoints

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming, `stream_options.include_usage`, tool calls, `n` (non-streaming) |
| `POST /v1/responses` | Responses API, streaming events, function calls |
| `POST /v1/completions` | legacy completions, streaming |
| `POST /v1/embeddings` | deterministic fake vectors (same text gives the same vector), `dimensions`, float/base64 |
| `GET /v1/models`, `GET /v1/models/{id}` | lists `settings.models` |

Errors use OpenAI's `{"error": {...}}` shape, so the SDK raises `RateLimitError`, `AuthenticationError` and so on as it would against the real API.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python server.py --port 8000
```

On first run, `server.py` copies `responses-sample.yaml` to `responses.yaml` if it doesn't exist. `responses.yaml` is your local, gitignored copy, so edit that one. Use `--config other.yaml` to load a different file.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
r = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])
print(r.choices[0].message.content)  # Hello! I'm a dummy assistant. ...
```

Most tools also pick it up from `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`.

## Configuring responses

Edit `responses.yaml` (start from [responses-sample.yaml](responses-sample.yaml)). Rules are checked top to bottom and the first match wins; if none match, `default` is used.

```yaml
rules:
  - name: jokes
    match:
      contains: [joke, funny]   # any substring (case-insensitive by default)
      # regex: '...'            # Python regex against the last user message
      # model: "gpt-4o-mini*"   # exact name, or prefix with trailing *
      # endpoint: chat          # chat | completions | responses
      # has_tools: true         # request included `tools`
    response:
      mode: cycle               # cycle (in order) or random, when content is a list
      content: ["joke one", "joke two"]
      # tool_calls: [{name: get_weather, arguments: {location: Paris}}]
      # finish_reason: length
      # latency_ms: 2000
      # error: {status: 429, type: requests, code: rate_limit_exceeded, message: "..."}
```

Content placeholders: `{{last_message}}`, `{{model}}`, `{{request_count}}`.

Global `settings`: `api_key` (require a bearer token), `models`, `strict_models`, `latency_ms`, `stream_chunk_delay_ms`, `error_rate`, `seed`, `embedding_dimensions`, `request_log_size`.

The sample config includes triggers you can type into any prompt: `[slow]`, `[ratelimit]`, `[error]`, `essay` (finish_reason `length`), `weather` with tools (tool call).

## Per-request overrides

These headers skip the rules:

- `X-Dummy-Response: some text` returns exactly that content.
- `X-Dummy-Error: 503` returns that HTTP error.

```python
client.chat.completions.create(..., extra_headers={"X-Dummy-Response": "forced reply"})
```

## Control endpoints

| Endpoint | Purpose |
|---|---|
| `POST /_dummy/rules` | add a rule at runtime (JSON body is a rule, checked before file rules) |
| `DELETE /_dummy/rules` | remove runtime rules |
| `GET /_dummy/requests` / `DELETE` | inspect or clear the log of received requests, with the rule that matched |
| `POST /_dummy/reload` | re-read the config file |
| `GET /_dummy/config` | show the active config |
| `GET /health` | liveness check |

## Limitations

Token counts are estimates (about 4 characters per token). Streaming always returns a single choice, whatever `n` is. Images, audio, files, assistants and batches are not implemented.
