from typing import Any

import requests
import os

PZ_MODEL_DATA_URL = (
    "https://palimpzest-research.s3.us-east-1.amazonaws.com/pz_models_information.json"
)


class Model:
    """
    Model describes the underlying LLM which should be used to perform some operation
    which requires invoking an LLM.
    """

    def __init__(self, model_id: str):
        self.metrics_manager = ModelMetricsManager()
        self.model_id = model_id
        # OpenRouter uses "google/" prefix; specs are stored under "gemini/" prefix
        lookup_id = model_id
        if model_id.startswith("google/"):
            lookup_id = "gemini/" + model_id[len("google/") :]
        self.model_specs = self.metrics_manager.get_model_metrics(lookup_id)

    def __lt__(self, other):
        if isinstance(other, Model):
            return self.value < other.value
        if isinstance(other, str):
            return self.value < other
        return NotImplemented

    @property
    def value(self) -> str:
        return self.model_id

    @property
    def provider(self) -> str | None:
        """Returns the provider string for this model."""
        return self.model_specs.get("provider")

    @property
    def api_key_env_var(self) -> str | None:
        """
        Returns the standard environment variable name for this provider's API key.
        """
        if self.provider == "gemini":
            return "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        mapping = {
            "openai": "OPENAI_API_KEY",
            "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
            "anthropic": "ANTHROPIC_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "hosted_vllm": "VLLM_API_KEY",
        }
        return mapping.get(self.provider)

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Model):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def is_llama_model(self) -> bool:
        return self.model_specs.get("is_llama_model", False)

    def is_embedding_model(self) -> bool:
        return self.model_specs.get("is_embedding_model", False)

    def is_text_image_multimodal_embedding_model(self) -> bool:
        return self.model_specs.get("is_text_image_multimodal_embedding_model", False)

    def is_provider_vertex_ai(self) -> bool:
        return self.provider == "vertex_ai"

    def is_provider_anthropic(self) -> bool:
        return self.provider == "anthropic"

    def is_provider_google_ai_studio(self) -> bool:
        return self.provider == "gemini" or self.provider == "google"

    def is_provider_openai(self) -> bool:
        return self.provider == "openai"

    def is_provider_together_ai(self) -> bool:
        return self.provider == "together_ai"

    def is_provider_deepseek(self) -> bool:
        return self.provider == "deepseek"

    def is_provider_ollama(self) -> bool:
        return self.provider == "ollama"

    def is_model_gemini(self) -> bool:
        return "gemini" in self.value.lower()

    def get_model_name(self) -> str:
        return self.value.split("/")[-1] if "/" in self.value else self.value

    def is_o_model(self) -> bool:
        return self.model_specs.get("is_o_model", False)

    def is_gpt_5_model(self) -> bool:
        return self.model_specs.get("is_gpt_5_model", False)

    def is_reasoning_model(self) -> bool:
        return self.model_specs.get("is_reasoning_model", False)

    def is_text_model(self) -> bool:
        return self.model_specs.get("is_text_model", False)

    def is_vision_model(self) -> bool:
        return self.model_specs.get("is_vision_model", False)

    def is_audio_model(self) -> bool:
        return self.model_specs.get("is_audio_model", False)

    def is_text_image_multimodal_model(self) -> bool:
        return self.is_text_model() and self.is_vision_model()

    def is_text_audio_multimodal_model(self) -> bool:
        return self.is_audio_model() and self.is_text_model()

    def supports_prompt_caching(self) -> bool:
        return (
            self.is_provider_anthropic()
            or self.is_provider_google_ai_studio()
            or self.is_provider_vertex_ai
            or self.is_provider_openai()
        ) and self.model_specs.get("supports_prompt_caching", False)

    def get_usd_per_input_token(self) -> float:
        return self.model_specs.get("usd_per_input_token", 0.0)

    def get_usd_per_audio_input_token(self) -> float:
        return self.model_specs.get(
            "usd_per_audio_input_token", self.get_usd_per_input_token()
        )

    # forward-looking, TODO: default value discussion
    def get_usd_per_image_input_token(self) -> float:
        return self.model_specs.get(
            "usd_per_image_input_token", self.get_usd_per_input_token()
        )

    def get_usd_per_cache_read_token(self) -> float:
        return self.model_specs.get(
            "usd_per_cache_read_token", self.get_usd_per_input_token()
        )

    def get_usd_per_audio_cache_read_token(self) -> float:
        return self.model_specs.get(
            "usd_per_audio_cache_read_token", self.get_usd_per_cache_read_token()
        )

    def get_usd_per_image_cache_read_token(self) -> float:
        return self.model_specs.get(
            "usd_per_image_cache_read_token", self.get_usd_per_cache_read_token()
        )

    # forward looking; Gemini explicit
    def get_usd_per_cached_token_per_hour(self) -> float:
        return self.model_specs.get("usd_per_cached_token_per_hour", 0.0)

    def get_usd_per_cache_creation_token(self) -> float:
        return self.model_specs.get("usd_per_cache_creation_token", 0.0)

    def get_usd_per_output_token(self) -> float:
        return self.model_specs.get("usd_per_output_token", 0.0)

    # forward-looking
    def get_usd_per_audio_cache_creation_token(self) -> float:
        return self.model_specs.get("usd_per_audio_cache_creation_token", 0.0)

    # forward-looking
    def get_usd_per_image_cache_creation_token(self) -> float:
        return self.model_specs.get("usd_per_image_cache_creation_token", 0.0)

    def get_seconds_per_output_token(self) -> float:
        return self.model_specs.get("seconds_per_output_token", 0.0)

    def get_overall_score(self) -> float:
        return self.model_specs.get("MMLU_Pro_score", 0.0)


class ModelMetricsManager:
    """
    Manages fetching and caching of model metrics from an external source.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.data_url = PZ_MODEL_DATA_URL
        self._metrics_cache = None
        self._initialized = True

    def _load_data(self):
        if self._metrics_cache is None:
            try:
                self._metrics_cache = requests.get(self.data_url).json()
            except Exception as e:
                self._metrics_cache = {}

    def get_model_metrics(self, model_name) -> dict[str, Any]:
        self._load_data()
        return self._metrics_cache.get(model_name, {})

    def refresh_data(self) -> None:
        self._metrics_cache = None
        self._load_data()
