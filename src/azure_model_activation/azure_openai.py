from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI, BadRequestError


@dataclass(frozen=True, slots=True)
class ActivationResult:
    text: str
    latency_ms: int
    usage: dict[str, int | None]
    raw: Any | None = None


def _get_usage_fields(resp: Any) -> dict[str, int | None]:
    usage_obj = getattr(resp, "usage", None)
    tokens_in = getattr(usage_obj, "prompt_tokens", None)
    tokens_out = getattr(usage_obj, "completion_tokens", None)
    return {
        "tokens_in": tokens_in if isinstance(tokens_in, int) else None,
        "tokens_out": tokens_out if isinstance(tokens_out, int) else None,
    }


def _looks_like_reasoning_only_chat_completion(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    msg = (choices[0] or {}).get("message")
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if content not in (None, ""):
        return False
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return False
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return False
    reasoning = details.get("reasoning_tokens")
    completion = usage.get("completion_tokens")
    return isinstance(reasoning, int) and isinstance(completion, int) and reasoning == completion


def _extract_text(resp: Any) -> str:
    def _from_content_obj(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text_val: Any = item.get("text") or item.get("content")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                    elif isinstance(text_val, dict):
                        nested = text_val.get("value") or text_val.get("content")
                        if isinstance(nested, str):
                            parts.append(nested)
                    continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
                    continue
                if text_attr is not None:
                    nested = getattr(text_attr, "value", None) or getattr(text_attr, "content", None)
                    if isinstance(nested, str):
                        parts.append(nested)
            return "".join(parts)
        return ""

    try:
        choice0 = resp.choices[0]
        msg = getattr(choice0, "message", None)
        if msg is None:
            return ""

        content = getattr(msg, "content", None)
        text = _from_content_obj(content)
        if text:
            return text

        refusal = getattr(msg, "refusal", None)
        if isinstance(refusal, str):
            return refusal

        if hasattr(resp, "model_dump"):
            dump = resp.model_dump()
            msg_dump = (
                (dump.get("choices") or [{}])[0].get("message") if isinstance(dump, dict) else None
            )
            if isinstance(msg_dump, dict):
                text = _from_content_obj(msg_dump.get("content"))
                if text:
                    return text
                refusal = msg_dump.get("refusal")
                if isinstance(refusal, str):
                    return refusal
    except Exception:
        return ""

    return ""


def _get_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def activate_model(
    *,
    deployment_name: str,
    prompt: str = "Hello! Reply with a short sentence.",
    endpoint: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 128,
    capture_raw: bool = False,
    api: str = "chat",
) -> ActivationResult:
    """Send a single chat completion to an Azure OpenAI deployment.

    This is intended as a lightweight "activation"/connectivity check.

    Args:
        deployment_name: Azure OpenAI deployment name (passed as `model=`).
        prompt: User prompt.
        endpoint: Base Azure endpoint, e.g. `https://<resource>.cognitiveservices.azure.com/`.
        api_key: Azure OpenAI API key.
        api_version: API version string.
    """

    resolved_endpoint = (endpoint or _get_env("AZURE_OPENAI_ENDPOINT") or _get_env("AZURE_OPENAI_API_BASE"))
    if not resolved_endpoint:
        raise ValueError("Missing Azure endpoint: set AZURE_OPENAI_ENDPOINT or pass endpoint=")

    resolved_key = api_key or _get_env("AZURE_OPENAI_API_KEY") or _get_env("AZURE_INFERENCE_CREDENTIAL")
    if not resolved_key:
        raise ValueError("Missing Azure API key: set AZURE_OPENAI_API_KEY or pass api_key=")

    resolved_version = api_version or _get_env("AZURE_OPENAI_API_VERSION")
    if not resolved_version:
        raise ValueError("Missing Azure API version: set AZURE_OPENAI_API_VERSION or pass api_version=")

    client = AzureOpenAI(
        azure_endpoint=resolved_endpoint.rstrip("/"),
        api_key=resolved_key,
        api_version=resolved_version,
    )

    api_norm = (api or "chat").strip().lower()
    if api_norm not in {"chat", "responses", "auto"}:
        raise ValueError("api must be one of: chat|responses|auto")

    def _run_chat(*, capture: bool) -> ActivationResult:
        body: dict[str, Any] = {
            "model": deployment_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(**body)
        except BadRequestError as exc:
            # Retry once with a compatible payload. Azure deployments can be strict about
            # token parameter names and may only support the default temperature.
            message = str(exc)
            retry_body = dict(body)

            # Token parameter name mismatch.
            if "Unsupported parameter: 'max_completion_tokens'" in message:
                retry_body.pop("max_completion_tokens", None)
                retry_body["max_tokens"] = max_tokens
            elif "Unsupported parameter: 'max_tokens'" in message:
                retry_body.pop("max_tokens", None)
                retry_body["max_completion_tokens"] = max_tokens

            # Some models/endpoints only support default temperature.
            if (
                "Unsupported value: 'temperature'" in message
                or "param': 'temperature'" in message
                or "param\": \"temperature\"" in message
            ):
                retry_body.pop("temperature", None)

            if retry_body == body:
                raise

            resp = client.chat.completions.create(**retry_body)

        latency_ms = int((time.perf_counter() - start) * 1000)
        raw = resp.model_dump() if (capture and hasattr(resp, "model_dump")) else (resp if capture else None)
        return ActivationResult(
            text=_extract_text(resp),
            latency_ms=latency_ms,
            usage=_get_usage_fields(resp),
            raw=raw,
        )

    def _run_responses(*, capture: bool) -> ActivationResult:
        start = time.perf_counter()
        try:
            resp = client.responses.create(
                model=deployment_name,
                input=prompt,
                max_output_tokens=max_tokens,
                reasoning={"effort": "minimal"},
                text={"verbosity": "low"},
            )
        except BadRequestError as exc:
            message = str(exc)
            if "Responses API is enabled only for api-version" in message:
                raise ValueError(
                    "This deployment requires a newer AZURE_OPENAI_API_VERSION for the Responses API. "
                    "Set AZURE_OPENAI_API_VERSION to 2025-03-01-preview or later (or pass api_version=), "
                    "or use api='chat'."
                ) from exc
            raise

        latency_ms = int((time.perf_counter() - start) * 1000)
        raw = resp.model_dump() if (capture and hasattr(resp, "model_dump")) else (resp if capture else None)

        text = getattr(resp, "output_text", None)
        if not isinstance(text, str):
            # Some providers don't populate output_text; extract from output items.
            collected: list[str] = []
            output_items = getattr(resp, "output", None)
            if isinstance(output_items, list):
                for item in output_items:
                    item_type = getattr(item, "type", None)
                    if item_type != "message":
                        continue
                    content_items = getattr(item, "content", None)
                    if not isinstance(content_items, list):
                        continue
                    for c in content_items:
                        c_type = getattr(c, "type", None)
                        c_text = getattr(c, "text", None)
                        if c_type == "output_text" and isinstance(c_text, str):
                            collected.append(c_text)
            elif isinstance(raw, dict):
                for item in raw.get("output") or []:
                    if not isinstance(item, dict) or item.get("type") != "message":
                        continue
                    for c in item.get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(
                            c.get("text"), str
                        ):
                            collected.append(c["text"])

            text = "".join(collected)

        usage_obj = getattr(resp, "usage", None)
        in_tokens = getattr(usage_obj, "input_tokens", None)
        out_tokens = getattr(usage_obj, "output_tokens", None)

        return ActivationResult(
            text=text,
            latency_ms=latency_ms,
            usage={
                "tokens_in": in_tokens if isinstance(in_tokens, int) else None,
                "tokens_out": out_tokens if isinstance(out_tokens, int) else None,
            },
            raw=raw,
        )

    if api_norm == "chat":
        return _run_chat(capture=capture_raw)

    if api_norm == "responses":
        return _run_responses(capture=capture_raw)

    # auto
    chat_result = _run_chat(capture=True)
    if chat_result.text.strip():
        if not capture_raw:
            return ActivationResult(
                text=chat_result.text,
                latency_ms=chat_result.latency_ms,
                usage=chat_result.usage,
                raw=None,
            )
        return chat_result

    if _looks_like_reasoning_only_chat_completion(chat_result.raw):
        # Try Responses API when chat completion returns reasoning-only.
        resp_result = _run_responses(capture=capture_raw)
        return resp_result

    # No useful content; return chat result (and drop raw unless requested)
    if capture_raw:
        return chat_result
    return ActivationResult(
        text=chat_result.text,
        latency_ms=chat_result.latency_ms,
        usage=chat_result.usage,
        raw=None,
    )
