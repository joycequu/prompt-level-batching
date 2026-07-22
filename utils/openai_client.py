"""
Direct client for OpenAI Responses API.

This module provides an OpenAIClient class that:
1. Calls the OpenAI Responses API directly (preferred over Chat Completions for
   reasoning models per https://developers.openai.com/api/docs/guides/reasoning)
2. Uses the same message format as the rest of the codebase (no transformation)
3. Relies on implicit prefix caching (automatic for supported models)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from utils.models import Model

logger = logging.getLogger(__name__)


@dataclass
class OpenAIResponse:
    """Response object matching GeminiResponse structure."""

    content: str
    usage: dict
    raw_response: Any = None
    thought: str = ""
    ttft: float | None = None


class OpenAIClient:
    """
    Direct client for OpenAI Responses API.
    Uses implicit prefix caching (automatic for GPT-4o+ and reasoning models).

    Args:
        model: Model object (e.g., Model("gpt-4o-2024-08-06"), Model("o4-mini"))
    """

    # Valid reasoning effort levels per the OpenAI docs
    OPENAI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

    # Reasoning model name prefixes/substrings (o1/o3/o4 series, gpt-5 series)
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")
    _REASONING_MODEL_SUBSTRINGS = ("gpt-5",)

    def __init__(self, model: Model):
        self.model = model
        self.model_name = model.get_model_name()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _is_reasoning_model(self) -> bool:
        name = self.model_name.lower()
        return any(name.startswith(p) for p in self._REASONING_MODEL_PREFIXES) or any(
            s in name for s in self._REASONING_MODEL_SUBSTRINGS
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        include_thoughts: bool = False,
        text_format: type | None = None,
        stream: bool = False,
    ) -> OpenAIResponse:
        """
        Generate content using OpenAI Responses API.

        Args:
            messages: List of messages in OpenAI format
            temperature: Sampling temperature (default: 0.0)
            reasoning_effort: Reasoning budget level — "none", "minimal", "low",
                "medium", "high", or "xhigh". Only applies to reasoning models.
            include_thoughts: If True, request a reasoning summary via
                reasoning.summary="auto" and return it as `thought`.
            text_format: Optional Pydantic model class for structured output.
                Uses client.responses.parse() so the SDK handles schema generation,
                additionalProperties, and strict mode automatically.

        Returns:
            OpenAIResponse with content, usage stats, raw response, and optional thought
        """
        if reasoning_effort is not None:
            if reasoning_effort not in self.OPENAI_REASONING_EFFORTS:
                raise ValueError(
                    f"Invalid reasoning_effort: {reasoning_effort!r}. "
                    f"Valid: {sorted(self.OPENAI_REASONING_EFFORTS)}"
                )
            if not self._is_reasoning_model():
                raise ValueError(
                    f"reasoning_effort is only supported for reasoning models "
                    f"(o1/o3/o4 series, gpt-5 series). "
                    f"Got model: {self.model_name!r}"
                )

        create_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            input=messages,
        )
        if not self._is_reasoning_model():
            create_kwargs["temperature"] = temperature

        # Build reasoning config: effort + optional summary for thought text
        reasoning_config: dict[str, Any] = {}
        if reasoning_effort is not None:
            reasoning_config["effort"] = reasoning_effort
        if include_thoughts and self._is_reasoning_model():
            reasoning_config["summary"] = "auto"
        if reasoning_config:
            create_kwargs["reasoning"] = reasoning_config

        # Use parse() with a Pydantic model — SDK handles schema, additionalProperties,
        # and strict mode automatically (per structured outputs docs).
        if stream:
            logger.warning(
                "Streaming is not supported for the OpenAI Responses API; "
                "falling back to non-streaming (ttft will be None)."
            )

        if text_format is not None:
            response = self.client.responses.parse(
                **create_kwargs,
                text_format=text_format,
            )
        else:
            response = self.client.responses.create(**create_kwargs)

        content = response.output_text or ""

        # Extract reasoning summary as thought text
        thought = ""
        if include_thoughts:
            for item in response.output:
                if getattr(item, "type", None) == "reasoning":
                    for summary_item in getattr(item, "summary", []):
                        thought += getattr(summary_item, "text", "")

        usage = self._extract_usage_stats(response)

        return OpenAIResponse(
            content=content,
            usage=usage,
            raw_response=response,
            thought=thought,
        )

    def _extract_usage_stats(self, response: Any) -> dict:
        """
        Extract usage statistics from OpenAI Responses API response.

        Responses API token field names differ from Chat Completions:
          input_tokens                           → total_prompt_tokens
          input_tokens_details.cached_tokens     → total_cache_read_tokens
          output_tokens                          → total_completion_tokens
          output_tokens_details.reasoning_tokens → reasoning_tokens
          output_tokens - reasoning_tokens       → output_text_tokens
        """
        generation_stats = {
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

        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return generation_stats

        usage_dict = (
            raw_usage.model_dump()
            if hasattr(raw_usage, "model_dump")
            else vars(raw_usage)
        )

        generation_stats["total_prompt_tokens"] = usage_dict.get("input_tokens", 0)
        generation_stats["total_completion_tokens"] = usage_dict.get("output_tokens", 0)

        input_details = usage_dict.get("input_tokens_details") or {}
        if isinstance(input_details, dict):
            cached = input_details.get("cached_tokens", 0)
            generation_stats["total_cache_read_tokens"] = cached
            generation_stats["text_cache_read_tokens"] = cached

        output_details = usage_dict.get("output_tokens_details") or {}
        if isinstance(output_details, dict):
            generation_stats["reasoning_tokens"] = output_details.get(
                "reasoning_tokens", 0
            )

        generation_stats["output_text_tokens"] = max(
            0,
            generation_stats["total_completion_tokens"]
            - generation_stats["reasoning_tokens"],
        )
        generation_stats["input_text_tokens"] = max(
            0,
            generation_stats["total_prompt_tokens"]
            - generation_stats["total_cache_read_tokens"],
        )

        generation_stats["cache_read_cost"] = (
            generation_stats["total_cache_read_tokens"]
            * self.model.get_usd_per_cache_read_token()
        )
        generation_stats["uncached_input_cost"] = (
            generation_stats["input_text_tokens"] * self.model.get_usd_per_input_token()
        )
        generation_stats["total_prompt_cost"] = (
            generation_stats["cache_read_cost"]
            + generation_stats["uncached_input_cost"]
        )
        generation_stats["total_completion_cost"] = (
            generation_stats["total_completion_tokens"]
            * self.model.get_usd_per_output_token()
        )
        generation_stats["total_cost"] = (
            generation_stats["total_prompt_cost"]
            + generation_stats["total_completion_cost"]
        )

        return generation_stats
