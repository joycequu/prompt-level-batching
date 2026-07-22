"""
Direct client for Gemini (Google AI Studio and Vertex AI) that bypasses litellm.

This module provides a GeminiClient class that:
1. Calls Gemini API directly via google-genai SDK
2. Converts litellm/palimpzest message format to Gemini format
3. Relies on implicit context caching (automatic prefix matching)
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from utils.models import Model

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Response object that mimics litellm completion response structure."""

    content: str
    usage: dict
    raw_response: Any = None
    thought: str = ""
    ttft: float | None = None


class GeminiClient:
    """
    Direct client for Gemini (Google AI Studio and Vertex AI) that bypasses litellm.
    Uses implicit caching (automatic prefix matching) for prompt caching.

    Uses a singleton pattern per (model, use_vertex) so that client state is shared
    across all Generator instances using the same model and provider.

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        use_vertex: If True, use Vertex AI; otherwise use Google AI Studio
    """

    _instances: dict[tuple[str, bool], GeminiClient] = {}

    # Gemini 2.5: maps reasoning_effort -> thinking_budget token count
    # Flash range: 0–24576, Pro range: 128–32768 (none=0 is Flash-only)
    # Reference: https://ai.google.dev/gemini-api/docs/thinking
    REASONING_EFFORT_TO_THINKING_BUDGET = {
        "none": 0,
        "minimal": 512,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
        "xhigh": 24576,
        "dynamic": -1,
    }

    # Gemini 3: uses thinkingLevel string; "none" and "dynamic" are not valid
    GEMINI_3_THINKING_LEVELS = {"minimal", "low", "medium", "high"}

    @classmethod
    def get_instance(cls, model: Model, use_vertex: bool = False) -> GeminiClient:
        """Get or create a singleton GeminiClient for the given model and provider."""
        key = (model.model_id, use_vertex)
        if key not in cls._instances:
            cls._instances[key] = cls(model, use_vertex)
        return cls._instances[key]

    def __init__(self, model: Model, use_vertex: bool = False):
        self.model = model
        self.model_name = model.get_model_name()
        self.use_vertex = use_vertex
        # Vertex AI: uses GOOGLE_APPLICATION_CREDENTIALS for auth
        self.client = genai.Client(vertexai=True) if use_vertex else genai.Client()

    def _is_gemini_3_model(self) -> bool:
        return "gemini-3" in self.model_name.lower()

    def _detect_image_media_type(self, base64_data: str) -> str:
        """Detect image format from base64 data by examining magic bytes."""
        try:
            header = base64.b64decode(base64_data[:32])
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                return "image/png"
            if header[:3] == b"\xff\xd8\xff":
                return "image/jpeg"
            if header[:6] in (b"GIF87a", b"GIF89a"):
                return "image/gif"
            if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                return "image/webp"
        except Exception:
            pass
        return "image/jpeg"

    def _transform_messages(
        self, messages: list[dict]
    ) -> tuple[str | None, list[dict]]:
        """
        Transform litellm/openrouter message format to Gemini API format.

        Args:
            messages: List of messages in litellm/palimpzest format

        Returns:
            Tuple of (system_instruction, gemini_contents)
        """
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            msg_type = msg.get("type")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    system_instruction = "".join(text_parts)
                else:
                    system_instruction = content

            elif role == "user":
                parts = []

                if msg_type == "text" or msg_type is None:
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                parts.append({"text": block.get("text", "")})
                    elif isinstance(content, str):
                        parts.append({"text": content})

                elif msg_type == "image":
                    for img in content:
                        if img.get("type") == "image_url":
                            url = img["image_url"]["url"]
                            if url.startswith("data:"):
                                # Robust parsing: handle "data:[<mediatype>];base64,<data>"
                                base64_marker = ";base64,"
                                marker_idx = url.find(base64_marker)
                                if marker_idx == -1:
                                    continue
                                data = url[marker_idx + len(base64_marker) :]
                                media_type = self._detect_image_media_type(data)
                                parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": media_type,
                                            "data": data,
                                        }
                                    }
                                )

                elif msg_type == "input_audio":
                    for audio in content:
                        if audio.get("type") == "input_audio":
                            audio_data = audio["input_audio"]
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": f"audio/{audio_data.get('format', 'wav')}",
                                        "data": audio_data["data"],
                                    }
                                }
                            )

                if parts:
                    # Merge consecutive user messages
                    if gemini_contents and gemini_contents[-1]["role"] == "user":
                        gemini_contents[-1]["parts"].extend(parts)
                    else:
                        gemini_contents.append({"role": "user", "parts": parts})

            elif role == "assistant":
                # Convert assistant to model role
                parts = []
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})

                if parts:
                    # Merge consecutive model messages (Gemini requires strict role alternation)
                    if gemini_contents and gemini_contents[-1]["role"] == "model":
                        gemini_contents[-1]["parts"].extend(parts)
                    else:
                        gemini_contents.append({"role": "model", "parts": parts})

        return system_instruction, gemini_contents

    def _extract_usage_stats(self, usage_metadata: Any) -> dict:
        """
        Extract and process usage statistics from Gemini response into the
        standard format expected by Generator.

        Args:
            usage_metadata: The usage_metadata from Gemini response

        Returns:
            Dictionary with information needed by GenerationStats.
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

        if usage_metadata is None:
            return generation_stats

        try:
            raw = usage_metadata.model_dump()
        except (AttributeError, Exception):
            # Fallback for SDK versions without model_dump()
            raw = vars(usage_metadata) if hasattr(usage_metadata, "__dict__") else {}
            logger.warning(
                "Could not call model_dump() on usage_metadata, using fallback"
            )

        generation_stats["total_cache_read_tokens"] = (
            raw.get("cached_content_token_count") or 0
        )

        # Parse cache read tokens by modality
        for detail in raw.get("cache_tokens_details") or []:
            modality = (detail.get("modality") or "").upper()
            token_count = detail.get("token_count") or 0
            if modality == "TEXT":
                generation_stats["text_cache_read_tokens"] = token_count
            elif modality == "IMAGE":
                generation_stats["image_cache_read_tokens"] = token_count
            elif modality == "AUDIO":
                generation_stats["audio_cache_read_tokens"] = token_count

        # Parse input tokens by modality (excludes cached tokens)
        for detail in raw.get("prompt_tokens_details") or []:
            modality = (detail.get("modality") or "").upper()
            token_count = detail.get("token_count") or 0
            if modality == "TEXT":
                generation_stats["input_text_tokens"] = max(
                    0, token_count - generation_stats["text_cache_read_tokens"]
                )
            elif modality == "IMAGE":
                generation_stats["input_image_tokens"] = max(
                    0, token_count - generation_stats["image_cache_read_tokens"]
                )
            elif modality == "AUDIO":
                generation_stats["input_audio_tokens"] = max(
                    0, token_count - generation_stats["audio_cache_read_tokens"]
                )

        generation_stats["reasoning_tokens"] = raw.get("thoughts_token_count") or 0
        generation_stats["output_text_tokens"] = raw.get("candidates_token_count") or 0

        generation_stats["total_prompt_tokens"] = raw.get("prompt_token_count") or 0
        generation_stats["total_cache_read_tokens"] = (
            raw.get("cached_content_token_count") or 0
        )
        generation_stats["total_completion_tokens"] = (
            generation_stats["reasoning_tokens"]
            + generation_stats["output_text_tokens"]
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

    def _build_generate_config(
        self,
        temperature: float,
        reasoning_effort: str | None,
        include_thoughts: bool,
        response_mime_type: str | None,
        response_json_schema: dict | None,
        system_instruction: str | None,
    ) -> types.GenerateContentConfig:
        config_kwargs: dict = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_json_schema is not None:
            config_kwargs["response_json_schema"] = response_json_schema

        if reasoning_effort is not None or include_thoughts:
            thinking_kwargs: dict = {}
            if include_thoughts:
                thinking_kwargs["include_thoughts"] = True
            if reasoning_effort is not None:
                if self._is_gemini_3_model():
                    if reasoning_effort == "disable":
                        raise ValueError(
                            "Gemini 3 does not support disabling thinking via reasoning_effort"
                        )
                    if reasoning_effort not in self.GEMINI_3_THINKING_LEVELS:
                        raise ValueError(
                            f"Invalid reasoning effort for Gemini 3: {reasoning_effort!r}. Valid: {self.GEMINI_3_THINKING_LEVELS}"
                        )
                    thinking_kwargs["thinking_level"] = reasoning_effort
                else:
                    budget = self.REASONING_EFFORT_TO_THINKING_BUDGET.get(reasoning_effort)
                    if budget is None:
                        raise ValueError(
                            f"Invalid reasoning effort: {reasoning_effort!r}. Valid: {list(self.REASONING_EFFORT_TO_THINKING_BUDGET)}"
                        )
                    thinking_kwargs["thinking_budget"] = budget
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

        return types.GenerateContentConfig(**config_kwargs)

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        include_thoughts: bool = False,
        response_mime_type: str | None = None,
        response_json_schema: dict | None = None,
        stream: bool = False,
    ) -> GeminiResponse:
        """
        Generate content using Gemini API directly.

        Args:
            messages: List of messages in openrouter/openai format
            temperature: Sampling temperature (default: 0.0)
            reasoning_effort: Optional thinking budget level — maps to thinking_level
                (Gemini 3: "minimal"/"low"/"medium"/"high") or thinking_budget (Gemini 2.5)
            include_thoughts: If True, extract and return thought text from thinking models
            response_mime_type: Optional MIME type for structured output, e.g. "application/json"
            response_json_schema: Optional Pydantic-derived JSON schema to constrain output shape
            stream: If True, use streaming to capture time-to-first-token (ttft)

        Returns:
            GeminiResponse with content, usage stats, raw response, and optional ttft
        """
        system_instruction, gemini_contents = self._transform_messages(messages)
        config = self._build_generate_config(
            temperature, reasoning_effort, include_thoughts,
            response_mime_type, response_json_schema, system_instruction,
        )

        if stream:
            return self._generate_stream(gemini_contents, config)

        response = self.client.models.generate_content(
            model=self.model_name, contents=gemini_contents, config=config,
        )

        content = ""
        thought = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thought += part.text
                else:
                    content += part.text

        usage = self._extract_usage_stats(response.usage_metadata)
        return GeminiResponse(
            content=content, usage=usage, raw_response=response, thought=thought,
        )

    def _generate_stream(
        self, gemini_contents: list, config: types.GenerateContentConfig
    ) -> GeminiResponse:
        """Streaming call that records time-to-first-token."""
        content = ""
        thought = ""
        ttft: float | None = None
        usage_metadata = None
        t0 = time.time()

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name, contents=gemini_contents, config=config
        ):
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if ttft is None:
                        ttft = time.time() - t0
                    if part.thought:
                        thought += part.text
                    else:
                        content += part.text
            if chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        usage = self._extract_usage_stats(usage_metadata)
        return GeminiResponse(content=content, usage=usage, thought=thought, ttft=ttft)
