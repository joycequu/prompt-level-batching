"""
Client for a locally-served vLLM instance (OpenAI-compatible Chat Completions API).

Connects to a vLLM server started with:
    python -m vllm.entrypoints.openai.api_server --model <model_id> ...

Cost fields are all zero — only token counts and latency are tracked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from utils.models import Model


@dataclass
class VLLMResponse:
    content: str
    usage: dict
    thought: str = ""
    ttft: float | None = None


class VLLMClient:
    """
    Thin wrapper around the vLLM OpenAI-compatible Chat Completions endpoint.

    Args:
        model: Model object — model.model_id is sent as the model name to vLLM
               (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        base_url: vLLM server base URL (default: http://localhost:8000/v1)
    """

    def __init__(self, model: Model, base_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        response_format: dict | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> VLLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model.model_id,
            messages=messages,
            temperature=temperature,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if stream:
            return self._generate_stream(kwargs)

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        return VLLMResponse(content=content, usage=usage)

    def _generate_stream(self, kwargs: dict) -> VLLMResponse:
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        parts: list[str] = []
        ttft: float | None = None
        final_chunk = None
        t0 = time.time()

        for chunk in self.client.chat.completions.create(**kwargs):
            if chunk.choices:
                text = getattr(chunk.choices[0].delta, "content", None) or ""
                if text and ttft is None:
                    ttft = time.time() - t0
                parts.append(text)
            if getattr(chunk, "usage", None) is not None:
                final_chunk = chunk

        content = "".join(parts)
        usage = self._extract_usage(final_chunk)
        return VLLMResponse(content=content, usage=usage, ttft=ttft)

    def _extract_usage(self, response: Any) -> dict:
        stats = {
            "total_prompt_tokens": 0,
            "input_text_tokens": 0,
            "input_image_tokens": 0,
            "input_audio_tokens": 0,
            "total_cache_read_tokens": 0,
            "text_cache_read_tokens": 0,
            "image_cache_read_tokens": 0,
            "audio_cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
            "output_text_tokens": 0,
            "total_completion_tokens": 0,
            "cache_read_cost": 0.0,
            "uncached_input_cost": 0.0,
            "total_prompt_cost": 0.0,
            "total_completion_cost": 0.0,
            "total_cost": 0.0,
        }
        if response is None or not hasattr(response, "usage") or response.usage is None:
            return stats

        u = response.usage
        prompt = getattr(u, "prompt_tokens", 0) or 0
        completion = getattr(u, "completion_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        stats["total_prompt_tokens"] = prompt
        stats["input_text_tokens"] = prompt
        stats["total_cache_read_tokens"] = cached
        stats["text_cache_read_tokens"] = cached
        stats["total_completion_tokens"] = completion
        stats["output_text_tokens"] = completion
        return stats
