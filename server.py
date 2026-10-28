"""Dummy OpenAI-compatible API server for development testing.

Run:  `python server.py --config responses.yaml --port 8000`
Point your OpenAI client at http://127.0.0.1:8000/v1 with any API key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "responses.yaml"
SAMPLE_CONFIG = HERE / "responses-sample.yaml"



# Configuration



class Match(BaseModel):
    """Conditions a request must satisfy for a rule to fire. All set fields must match."""

    contains: str | list[str] | None = None  # any of these substrings
    regex: str | None = None
    model: str | None = None  # exact model name, or prefix ending in "*"
    endpoint: Literal["chat", "completions", "responses"] | None = None
    has_tools: bool | None = None
    case_sensitive: bool = False


class ErrorSpec(BaseModel):
    status: int = 500
    message: str = "Simulated error from dummy server"
    type: str = "server_error"
    code: str | None = None


class ToolCallSpec(BaseModel):
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class ResponseSpec(BaseModel):
    """What to return. `content` may be one string or a list picked by `mode`.

    Content supports placeholders: {{last_message}}, {{model}}, {{request_count}}.
    """

    content: str | list[str] | None = None
    mode: Literal["cycle", "random"] = "cycle"
    tool_calls: list[ToolCallSpec] | None = None
    finish_reason: str | None = None
    error: ErrorSpec | None = None
    latency_ms: int | None = None


class Rule(BaseModel):
    name: str | None = None
    match: Match = Field(default_factory=Match)
    response: ResponseSpec


class Settings(BaseModel):
    api_key: str | None = None  # when set, requests must send "Authorization: Bearer <api_key>"
    models: list[str] = Field(
        default_factory=lambda: ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "text-embedding-3-small"]
    )
    strict_models: bool = False  # reject models not in `models` with 404
    latency_ms: int = 0  # delay before every response
    stream_chunk_delay_ms: int = 15  # delay between streamed chunks
    error_rate: float = 0.0  # probability (0-1) of a random 500 on any generation request
    seed: int | None = None  # seed for "random" mode and error_rate
    embedding_dimensions: int = 1536
    request_log_size: int = 200


class Config(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    rules: list[Rule] = Field(default_factory=list)
    default: ResponseSpec = Field(
        default_factory=lambda: ResponseSpec(
            content="This is a dummy response from the fake OpenAI server. You said: {{last_message}}"
        )
    )


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return Config.model_validate(data or {})



# Rule matching and response selection



_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Resolved:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    error: ErrorSpec | None = None
    latency_ms: int = 0
    rule: str | None = None


def count_tokens(text: str | None) -> int:
    """Rough token estimate (~4 chars per token); good enough for fake usage numbers."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


class Responder:
    def __init__(self, config: Config):
        self._lock = threading.Lock()
        self.load(config)

    def load(self, config: Config) -> None:
        with self._lock:
            self.config = config
            self.runtime_rules: list[Rule] = []
            self._cycle: dict[int, int] = {}
            self._rng = random.Random(config.settings.seed)
            self.request_count = 0
            self.request_log: deque[dict[str, Any]] = deque(maxlen=config.settings.request_log_size)

    @property
    def settings(self):
        return self.config.settings

    def add_rule(self, rule: Rule) -> None:
        with self._lock:
            self.runtime_rules.insert(0, rule)

    def clear_runtime_rules(self) -> None:
        with self._lock:
            self.runtime_rules.clear()

    def log_request(self, endpoint: str, body: dict[str, Any], rule: str | None) -> None:
        with self._lock:
            self.request_log.append(
                {"time": time.time(), "endpoint": endpoint, "rule": rule, "body": body}
            )

    def resolve(
        self,
        *,
        endpoint: str,
        model: str,
        text: str,
        has_tools: bool = False,
        override_content: str | None = None,
        override_error: int | None = None,
    ) -> Resolved:
        with self._lock:
            self.request_count += 1
            count = self.request_count
            latency = self.settings.latency_ms

            if override_error is not None:
                return Resolved(None, error=ErrorSpec(status=override_error), latency_ms=latency,
                                rule="header:x-dummy-error")
            if override_content is not None:
                return Resolved(override_content, latency_ms=latency, rule="header:x-dummy-response")
            if self.settings.error_rate and self._rng.random() < self.settings.error_rate:
                return Resolved(None, error=ErrorSpec(), latency_ms=latency, rule="error_rate")

            rule = next(
                (r for r in [*self.runtime_rules, *self.config.rules]
                 if _matches(r, endpoint, model, text, has_tools)),
                None,
            )
            spec = rule.response if rule else self.config.default
            rule_name = (rule.name or "unnamed") if rule else "default"

            content = self._pick_content(spec)
            if content is not None:
                content = _render(content, {"last_message": text, "model": model, "request_count": count})

            tool_calls = [
                ToolCall(
                    id=new_id("call_"),
                    name=tc.name,
                    arguments=tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments),
                )
                for tc in spec.tool_calls or []
            ]
            finish = spec.finish_reason or ("tool_calls" if tool_calls else "stop")
            return Resolved(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish,
                error=spec.error,
                latency_ms=spec.latency_ms if spec.latency_ms is not None else latency,
                rule=rule_name,
            )

    def _pick_content(self, spec: ResponseSpec) -> str | None:
        if spec.content is None or isinstance(spec.content, str):
            return spec.content
        if not spec.content:
            return ""
        if spec.mode == "random":
            return self._rng.choice(spec.content)
        key = id(spec)
        i = self._cycle.get(key, 0)
        self._cycle[key] = i + 1
        return spec.content[i % len(spec.content)]


def _matches(rule: Rule, endpoint: str, model: str, text: str, has_tools: bool) -> bool:
    m = rule.match
    if m.endpoint and m.endpoint != endpoint:
        return False
    if m.model:
        if m.model.endswith("*"):
            if not model.startswith(m.model[:-1]):
                return False
        elif m.model != model:
            return False
    if m.has_tools is not None and m.has_tools != has_tools:
        return False
    haystack = text if m.case_sensitive else text.lower()
    if m.contains is not None:
        needles = [m.contains] if isinstance(m.contains, str) else m.contains
        if not m.case_sensitive:
            needles = [n.lower() for n in needles]
        if not any(n in haystack for n in needles):
            return False
    if m.regex is not None:
        flags = 0 if m.case_sensitive else re.IGNORECASE
        if not re.search(m.regex, text, flags | re.DOTALL):
            return False
    return True


def _render(template: str, values: dict[str, Any]) -> str:
    return _PLACEHOLDER.sub(lambda mt: str(values.get(mt.group(1), mt.group(0))), template)



# HTTP app



_WORDS = re.compile(r"\s*\S+\s*|\s+")


def create_app(config_path: str | Path | None = None, config: Config | None = None) -> FastAPI:
    app = FastAPI(title="Dummy OpenAI API", version="0.1.0")
    responder = Responder(config or load_config(config_path))
    app.state.responder = responder

    # helpers

    def error(status: int, message: str, type_: str, code: str | None = None, param: str | None = None):
        return JSONResponse(
            status_code=status,
            content={"error": {"message": message, "type": type_, "param": param, "code": code}},
        )

    def check_request(request: Request, model: str | None, require_model: bool = True):
        s = responder.settings
        if s.api_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {s.api_key}":
                return error(401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key")
        if require_model and not model:
            return error(400, "you must provide a model parameter", "invalid_request_error", param="model")
        if model and s.strict_models and model not in s.models:
            return error(404, f"The model `{model}` does not exist or you do not have access to it.",
                         "invalid_request_error", "model_not_found")
        return None

    async def read_json(request: Request) -> dict[str, Any] | JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return error(400, "We could not parse the JSON body of your request.", "invalid_request_error")
        if not isinstance(body, dict):
            return error(400, "Request body must be a JSON object.", "invalid_request_error")
        return body

    def resolve(request: Request, endpoint: str, body: dict, text: str) -> Resolved:
        err_header = request.headers.get("x-dummy-error")
        result = responder.resolve(
            endpoint=endpoint,
            model=body.get("model", ""),
            text=text,
            has_tools=bool(body.get("tools")),
            override_content=request.headers.get("x-dummy-response"),
            override_error=int(err_header) if err_header and err_header.isdigit() else None,
        )
        responder.log_request(endpoint, body, result.rule)
        return result

    def error_from(result: Resolved):
        e = result.error
        return error(e.status, e.message, e.type, e.code)

    async def delay(ms: int):
        if ms > 0:
            await asyncio.sleep(ms / 1000)

    def chunk_text(text: str) -> list[str]:
        return _WORDS.findall(text) or [text]

    def sse(data: Any, event: str | None = None) -> str:
        payload = data if isinstance(data, str) else json.dumps(data)
        return (f"event: {event}\n" if event else "") + f"data: {payload}\n\n"

    def stream(gen: AsyncIterator[str]) -> StreamingResponse:
        return StreamingResponse(gen, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # models

    def model_obj(name: str) -> dict:
        return {"id": name, "object": "model", "created": 1700000000, "owned_by": "dummy-openai"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        if err := check_request(request, None, require_model=False):
            return err
        return {"object": "list", "data": [model_obj(m) for m in responder.settings.models]}

    @app.get("/v1/models/{model:path}")
    async def get_model(model: str, request: Request):
        if err := check_request(request, None, require_model=False):
            return err
        if model not in responder.settings.models:
            return error(404, f"The model '{model}' does not exist", "invalid_request_error", "model_not_found")
        return model_obj(model)

    # chat completions

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await read_json(request)
        if isinstance(body, JSONResponse):
            return body
        model = body.get("model")
        if err := check_request(request, model):
            return err
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return error(400, "'messages' must be a non-empty array", "invalid_request_error", param="messages")

        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        last = extract_text(user_msgs[-1].get("content")) if user_msgs else ""
        prompt_tokens = sum(count_tokens(extract_text(m.get("content"))) + 4
                            for m in messages if isinstance(m, dict))
        n = max(1, int(body.get("n") or 1))
        cid, created = new_id("chatcmpl-"), int(time.time())

        if body.get("stream"):
            result = resolve(request, "chat", body, last)
            await delay(result.latency_ms)
            if result.error:
                return error_from(result)
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return stream(stream_chat(result, cid, created, model, prompt_tokens, include_usage))

        results = [resolve(request, "chat", body, last) for _ in range(n)]
        await delay(results[0].latency_ms)
        if results[0].error:
            return error_from(results[0])

        choices = []
        for i, r in enumerate(results):
            message: dict[str, Any] = {"role": "assistant", "content": r.content, "refusal": None}
            if r.tool_calls:
                message["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in r.tool_calls
                ]
            choices.append({"index": i, "message": message, "logprobs": None, "finish_reason": r.finish_reason})
        completion_tokens = sum(completion_token_count(r) for r in results)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_dummy",
            "choices": choices,
            "usage": usage(prompt_tokens, completion_tokens),
        }

    async def stream_chat(result: Resolved, cid: str, created: int, model: str,
                          prompt_tokens: int, include_usage: bool) -> AsyncIterator[str]:
        delay_ms = responder.settings.stream_chunk_delay_ms

        def chunk(delta: dict, finish: str | None = None) -> str:
            data = {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "system_fingerprint": "fp_dummy",
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
            }
            if include_usage:
                data["usage"] = None
            return sse(data)

        yield chunk({"role": "assistant", "content": "", "refusal": None})
        if result.content:
            for piece in chunk_text(result.content):
                await delay(delay_ms)
                yield chunk({"content": piece})
        for i, tc in enumerate(result.tool_calls):
            await delay(delay_ms)
            yield chunk({"tool_calls": [{"index": i, "id": tc.id, "type": "function",
                                         "function": {"name": tc.name, "arguments": ""}}]})
            for piece in split_fixed(tc.arguments, 16):
                yield chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
        yield chunk({}, result.finish_reason)
        if include_usage:
            yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "system_fingerprint": "fp_dummy", "choices": [],
                       "usage": usage(prompt_tokens, completion_token_count(result))})
        yield sse("[DONE]")

    # legacy completions

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await read_json(request)
        if isinstance(body, JSONResponse):
            return body
        model = body.get("model")
        if err := check_request(request, model):
            return err
        prompt = body.get("prompt", "")
        prompt = "\n".join(map(str, prompt)) if isinstance(prompt, list) else str(prompt)
        result = resolve(request, "completions", body, prompt)
        await delay(result.latency_ms)
        if result.error:
            return error_from(result)
        text = result.content or ""
        cid, created = new_id("cmpl-"), int(time.time())
        finish = "length" if result.finish_reason == "length" else "stop"

        if body.get("stream"):
            async def gen():
                for piece in chunk_text(text):
                    await delay(responder.settings.stream_chunk_delay_ms)
                    yield sse({"id": cid, "object": "text_completion", "created": created, "model": model,
                               "choices": [{"index": 0, "text": piece, "logprobs": None, "finish_reason": None}]})
                yield sse({"id": cid, "object": "text_completion", "created": created, "model": model,
                           "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": finish}]})
                yield sse("[DONE]")
            return stream(gen())

        return {
            "id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
            "usage": usage(count_tokens(prompt), count_tokens(text)),
        }

    # responses API

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await read_json(request)
        if isinstance(body, JSONResponse):
            return body
        model = body.get("model")
        if err := check_request(request, model):
            return err
        inp = body.get("input", "")
        if isinstance(inp, str):
            last, all_text = inp, inp
        else:
            items = [i for i in inp if isinstance(i, dict)] if isinstance(inp, list) else []
            users = [i for i in items if i.get("role") == "user"]
            last = extract_text(users[-1].get("content")) if users else ""
            all_text = " ".join(extract_text(i.get("content")) for i in items)
        input_tokens = count_tokens(all_text) + count_tokens(body.get("instructions"))

        result = resolve(request, "responses", body, last)
        await delay(result.latency_ms)
        if result.error:
            return error_from(result)

        rid, msg_id, created = new_id("resp_"), new_id("msg_"), int(time.time())
        output: list[dict] = []
        if result.content is not None:
            output.append({
                "type": "message", "id": msg_id, "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": result.content, "annotations": []}],
            })
        for tc in result.tool_calls:
            output.append({"type": "function_call", "id": new_id("fc_"), "call_id": tc.id,
                           "name": tc.name, "arguments": tc.arguments, "status": "completed"})

        def response_obj(status: str, out: list[dict], with_usage: bool) -> dict:
            out_tokens = completion_token_count(result)
            return {
                "id": rid, "object": "response", "created_at": created, "status": status,
                "model": model, "output": out, "error": None, "incomplete_details": None,
                "instructions": body.get("instructions"), "metadata": body.get("metadata") or {},
                "parallel_tool_calls": True, "temperature": body.get("temperature", 1.0),
                "top_p": body.get("top_p", 1.0), "tool_choice": body.get("tool_choice", "auto"),
                "tools": body.get("tools") or [],
                "previous_response_id": body.get("previous_response_id"),
                "usage": {
                    "input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": out_tokens, "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": input_tokens + out_tokens,
                } if with_usage else None,
            }

        if not body.get("stream"):
            return response_obj("completed", output, True)

        async def gen():
            seq = 0

            def ev(type_: str, **data) -> str:
                nonlocal seq
                seq += 1
                return sse({"type": type_, "sequence_number": seq - 1, **data}, event=type_)

            delay_ms = responder.settings.stream_chunk_delay_ms
            yield ev("response.created", response=response_obj("in_progress", [], False))
            yield ev("response.in_progress", response=response_obj("in_progress", [], False))
            for idx, item in enumerate(output):
                if item["type"] == "message":
                    text = item["content"][0]["text"]
                    yield ev("response.output_item.added", output_index=idx,
                             item={**item, "status": "in_progress", "content": []})
                    yield ev("response.content_part.added", item_id=item["id"], output_index=idx,
                             content_index=0, part={"type": "output_text", "text": "", "annotations": []})
                    for piece in chunk_text(text) if text else []:
                        await delay(delay_ms)
                        yield ev("response.output_text.delta", item_id=item["id"], output_index=idx,
                                 content_index=0, delta=piece, logprobs=[])
                    yield ev("response.output_text.done", item_id=item["id"], output_index=idx,
                             content_index=0, text=text, logprobs=[])
                    yield ev("response.content_part.done", item_id=item["id"], output_index=idx,
                             content_index=0, part=item["content"][0])
                else:
                    yield ev("response.output_item.added", output_index=idx,
                             item={**item, "status": "in_progress", "arguments": ""})
                    for piece in split_fixed(item["arguments"], 16):
                        await delay(delay_ms)
                        yield ev("response.function_call_arguments.delta", item_id=item["id"],
                                 output_index=idx, delta=piece)
                    yield ev("response.function_call_arguments.done", item_id=item["id"],
                             output_index=idx, arguments=item["arguments"])
                yield ev("response.output_item.done", output_index=idx, item=item)
            yield ev("response.completed", response=response_obj("completed", output, True))

        return stream(gen())

    # embeddings

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body = await read_json(request)
        if isinstance(body, JSONResponse):
            return body
        model = body.get("model")
        if err := check_request(request, model):
            return err
        inp = body.get("input")
        if isinstance(inp, str) or (isinstance(inp, list) and inp and isinstance(inp[0], int)):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            return error(400, "'input' must be a non-empty string or array", "invalid_request_error",
                         param="input")
        dims = int(body.get("dimensions") or responder.settings.embedding_dimensions)
        b64 = body.get("encoding_format") == "base64"
        await delay(responder.settings.latency_ms)
        responder.log_request("embeddings", body, None)

        data, tokens = [], 0
        for i, item in enumerate(inp):
            text = item if isinstance(item, str) else json.dumps(item)
            tokens += len(item) if isinstance(item, list) else count_tokens(text)
            vec = fake_embedding(text, dims)
            emb = base64.b64encode(struct.pack(f"<{dims}f", *vec)).decode() if b64 else vec
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return {"object": "list", "data": data, "model": model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    # dev/admin endpoints

    @app.get("/_dummy/config")
    async def get_config():
        return {**responder.config.model_dump(),
                "runtime_rules": [r.model_dump() for r in responder.runtime_rules]}

    @app.post("/_dummy/reload")
    async def reload_config():
        if config_path is None:
            return error(400, "Server was started without a config file", "invalid_request_error")
        responder.load(load_config(config_path))
        return {"status": "reloaded", "rules": len(responder.config.rules)}

    @app.post("/_dummy/rules")
    async def add_rule(rule: Rule):
        responder.add_rule(rule)
        return {"status": "added", "runtime_rules": len(responder.runtime_rules)}

    @app.delete("/_dummy/rules")
    async def clear_rules():
        responder.clear_runtime_rules()
        return {"status": "cleared"}

    @app.get("/_dummy/requests")
    async def get_requests():
        return {"count": len(responder.request_log), "requests": list(responder.request_log)}

    @app.delete("/_dummy/requests")
    async def clear_requests():
        responder.request_log.clear()
        return {"status": "cleared"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def extract_text(content: Any) -> str:
    """Flatten OpenAI message content (string or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return str(content)


def completion_token_count(r: Resolved) -> int:
    return count_tokens(r.content) + sum(count_tokens(tc.name + tc.arguments) for tc in r.tool_calls)


def usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0,
                                      "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
    }


def split_fixed(s: str, size: int) -> list[str]:
    return [s[i:i + size] for i in range(0, len(s), size)] or [""]


def fake_embedding(text: str, dims: int) -> list[float]:
    """Deterministic unit vector derived from the text, so equal inputs give equal embeddings."""
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]



# Entry point



def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dummy OpenAI-compatible API server.")
    parser.add_argument("-c", "--config", default=os.environ.get("DUMMY_OPENAI_CONFIG", str(DEFAULT_CONFIG)),
                        help="Path to a YAML/JSON responses config (env: DUMMY_OPENAI_CONFIG)")
    parser.add_argument("--host", default=os.environ.get("DUMMY_OPENAI_HOST", "127.0.0.1"))
    parser.add_argument("-p", "--port", type=int, default=int(os.environ.get("DUMMY_OPENAI_PORT", "8000")))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and config_path == DEFAULT_CONFIG and SAMPLE_CONFIG.exists():
        shutil.copyfile(SAMPLE_CONFIG, config_path)
        print(f"Created {config_path.name} from {SAMPLE_CONFIG.name}; edit it to change responses.")

    uvicorn.run(create_app(config_path), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
