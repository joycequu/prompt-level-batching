"""
Direct client for OpenRouter API.

OpenRouter API Documentation: https://openrouter.ai/docs/api/reference/overview

This module handles:
- OpenRouter API request formatting
- Response parsing and usage statistics extraction
- Cost tracking per call
- Prompt caching transformations (Anthropic cache_control)
"""

import copy
import os
import time
from typing import Any, Dict, List
from dataclasses import dataclass

from openai import OpenAI

from utils.models import Model


@dataclass
class OpenRouterResponse:
    """Response object matching GeminiResponse structure."""

    content: str
    usage: dict
    raw_response: Any = None
    thought: str = ""
    ttft: float | None = None


class OpenRouterClient:
    """
    Manages OpenRouter API requests and response parsing.

    This class handles:
    1. Request formatting for OpenRouter API
    2. Response parsing and usage extraction per call
    3. Prompt caching transformations (Anthropic cache_control)
    """

    CACHE_BOUNDARY_MARKER = "<<cache-boundary>>"

    def __init__(self, model: Model):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def _is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic model"""
        return self.model.model_id.startswith("anthropic/")

    def get_request_kwargs(
        self,
        provider_preferences: Dict[str, Any] | None = None,
        transforms: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Get OpenRouter-specific request kwargs for the API call.

        Args:
            provider_preferences: Optional provider-specific preferences
                e.g., {"allow_fallbacks": False, "order": ["Anthropic", "Google"]}
            transforms: Optional list of transforms
                e.g., ["middle-out"] for prompt compression

        Returns:
            Dictionary of kwargs to pass to chat.completions.create()
        """
        extra_body = {}

        if provider_preferences:
            extra_body["provider"] = provider_preferences

        if transforms:
            extra_body["transforms"] = transforms

        return {"extra_body": extra_body} if extra_body else {}

    def _build_create_kwargs(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Dict[str, Any] | None,
        reasoning_effort: str | None,
        disable_cache: bool,
        max_tokens: int | None = None,
    ) -> Dict[str, Any]:
        transformed_messages = (
            messages if disable_cache else self.transform_messages_for_caching(messages)
        )
        create_kwargs: Dict[str, Any] = dict(
            model=self.model.model_id,
            messages=transformed_messages,
            temperature=temperature,
            **self.get_request_kwargs(),
        )
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        if reasoning_effort is not None:
            _OPENROUTER_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
            if reasoning_effort not in _OPENROUTER_VALID_EFFORTS:
                raise ValueError(
                    f"Invalid reasoning_effort for OpenRouter: {reasoning_effort!r}. "
                    f"Valid: {sorted(_OPENROUTER_VALID_EFFORTS)}"
                )
            extra_body = create_kwargs.setdefault("extra_body", {})
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        return create_kwargs

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        response_format: Dict[str, Any] | None = None,
        disable_cache: bool = False,
        reasoning_effort: str | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> OpenRouterResponse:
        """
        Generate content using OpenRouter API.

        Args:
            messages: List of messages in litellm/palimpzest format
            temperature: Sampling temperature (default: 0.0)
            response_format: Optional response format dict, e.g. {"type": "json_object"}
            disable_cache: If True, skip cache_control transforms so no explicit caching is requested
            stream: If True, use streaming to capture time-to-first-token (ttft)
            max_tokens: If set, cap the number of output tokens

        Returns:
            OpenRouterResponse with content, usage stats, raw response, and optional ttft
        """
        create_kwargs = self._build_create_kwargs(
            messages, temperature, response_format, reasoning_effort, disable_cache, max_tokens
        )
        if stream:
            return self._generate_stream(create_kwargs)

        response = self.client.chat.completions.create(**create_kwargs)
        usage = self._extract_usage_stats(response)
        message = response.choices[0].message
        content = message.content or ""
        thought = getattr(message, "reasoning", None) or ""
        return OpenRouterResponse(
            content=content, usage=usage, raw_response=response, thought=thought
        )

    def _generate_stream(self, create_kwargs: Dict[str, Any]) -> OpenRouterResponse:
        """Streaming call that records time-to-first-token."""
        create_kwargs = {
            **create_kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        content_parts: List[str] = []
        thought_parts: List[str] = []
        ttft: float | None = None
        final_chunk = None
        t0 = time.time()

        for chunk in self.client.chat.completions.create(**create_kwargs):
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                reasoning = getattr(delta, "reasoning", None) or ""
                if (text or reasoning) and ttft is None:
                    ttft = time.time() - t0
                if text:
                    content_parts.append(text)
                if reasoning:
                    thought_parts.append(reasoning)
            if getattr(chunk, "usage", None) is not None:
                final_chunk = chunk

        content = "".join(content_parts)
        thought = "".join(thought_parts)
        usage = self._extract_usage_stats(final_chunk)
        return OpenRouterResponse(content=content, usage=usage, thought=thought, ttft=ttft)

    def _extract_usage_stats(self, response: Any) -> dict:
        """
        Extract usage statistics from OpenRouter API response.

        Returns a dict with field names consistent with GeminiClient._extract_usage_stats():
            input_text_tokens, output_text_tokens, total_cache_read_tokens,
            cache_creation_tokens, reasoning_tokens, plus OpenRouter cost fields.

        Args:
            response: OpenAI/OpenRouter completion response object

        Returns:
            Usage dict with standard field names.
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
            "uncached_input_cost": 0.0,  # excludes cached
            "total_prompt_cost": 0.0,
            "total_completion_cost": 0.0,
            "total_cost": 0.0,
        }

        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            else:
                usage_dict = dict(usage) if usage else {}
        else:
            usage_dict = {}

        generation_stats["total_prompt_tokens"] = usage_dict.get("prompt_tokens", 0)
        generation_stats["total_completion_tokens"] = usage_dict.get(
            "completion_tokens", 0
        )

        # Extract reasoning tokens from completion_tokens_details
        completion_details = usage_dict.get("completion_tokens_details", {})
        if isinstance(completion_details, dict):
            generation_stats["reasoning_tokens"] = completion_details.get(
                "reasoning_tokens", 0
            )

        generation_stats["output_text_tokens"] = max(
            0,
            generation_stats["total_completion_tokens"]
            - generation_stats["reasoning_tokens"],
        )

        # Extract cache read tokens from prompt_tokens_details
        prompt_details = usage_dict.get("prompt_tokens_details", {})
        if isinstance(prompt_details, dict):
            generation_stats["total_cache_read_tokens"] = prompt_details.get(
                "cached_tokens", 0
            )
            generation_stats["cache_creation_tokens"] = prompt_details.get(
                "cache_write_tokens", 0
            )
            generation_stats["input_audio_tokens"] = prompt_details.get(
                "audio_tokens", 0
            )

        # input_text_tokens = uncached portion of prompt
        generation_stats["input_text_tokens"] = max(
            0,
            generation_stats["total_prompt_tokens"]
            - generation_stats["total_cache_read_tokens"]
            - generation_stats["input_audio_tokens"],
        )

        # Extract cost
        generation_stats["total_cost"] = float(usage_dict.get("cost", 0.0))
        cost_details = usage_dict.get("cost_details", {})
        if isinstance(cost_details, dict):
            generation_stats["total_prompt_cost"] = float(
                cost_details.get("upstream_inference_prompt_cost", 0.0)
            )
            generation_stats["total_completion_cost"] = float(
                cost_details.get("upstream_inference_completions_cost", 0.0)
            )

        return generation_stats

    def transform_messages_for_caching(self, messages: List[Dict]) -> List[Dict]:
        """
        Transform messages to enable prompt caching for supported providers.

        For Anthropic models via OpenRouter:
        - Adds cache_control markers to system messages
        - Splits user messages at <<cache-boundary>> marker into cached/uncached blocks

        For other providers:
        - Returns messages unchanged (OpenRouter may handle caching automatically)
        """
        if self._is_anthropic_model():
            return self._transform_messages_for_anthropic(messages)
        return messages

    def _transform_messages_for_anthropic(self, messages: List[Dict]) -> List[Dict]:
        """
        Add cache_control markers to messages for Anthropic models.

        Transforms messages to:
        1. Add cache_control to system message content blocks
        2. Convert user messages with <<cache-boundary>> marker into multiple content blocks:
            a. Static prefix block (with cache_control) - cacheable across requests
            b. Dynamic content block (without cache_control) - changes per request
        """
        result = []
        for message in messages:
            new_message = copy.deepcopy(message)
            role = new_message.get("role")
            content = new_message.get("content", "")

            if role == "system":
                if isinstance(content, str) and content:
                    new_message["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif isinstance(content, list) and content:
                    last_block = new_message["content"][-1]
                    if (
                        isinstance(last_block, dict)
                        and last_block.get("type") == "text"
                    ):
                        last_block["cache_control"] = {"type": "ephemeral"}

            elif (
                role == "user"
                and isinstance(content, str)
                and self.CACHE_BOUNDARY_MARKER in content
            ):
                static, dynamic = content.split(self.CACHE_BOUNDARY_MARKER, 1)

                new_blocks = []
                if static.strip():
                    new_blocks.append(
                        {
                            "type": "text",
                            "text": static,
                            "cache_control": {"type": "ephemeral"},
                        }
                    )

                if dynamic.strip():
                    new_blocks.append({"type": "text", "text": dynamic})

                new_message["content"] = new_blocks if new_blocks else ""

            result.append(new_message)
        return result
