"""
BatchedExecutor — abstract base class for document-batching LLM experiments.

The base class owns the full execution loop and all shared infrastructure.
Subclasses implement three small hooks:

  _get_system_prompt() -> str
      Return the system prompt for this pass. Called once per execute().

  _process_batch(batch, output, thought, usage, duration, state) -> dict | None
      Parse LLM output for one batch, update state["results"], write per-call CSV.
      Return a tqdm postfix dict or None for defaults.

  _finalize(state, docs) -> Any   [optional]
      Reshape state["results"] before returning from execute(). Default: pass-through.

For multi-pass tasks (e.g. looping over 41 CUAD fields), override execute() and
call _run_batch_loop() once per pass.

The existing executors (BatchedCUADExecutor, BatchedFilterExecutor) pre-date this
base class and are not subclasses, but share the same structure.
"""

import csv
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models import Model
from utils.gemini_client import GeminiClient
from utils.openai_client import OpenAIClient
from utils.openrouter_client import OpenRouterClient
from utils.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


# Fields that every per-call CSV row contains, regardless of task.
# Domain-specific lists should extend this with their own prefix/suffix fields.
SHARED_PER_CALL_FIELDNAMES = [
    "run_id",
    "batch_number",
    "latency_secs",
    "ttft_secs",
    "input_text_tokens",
    "total_cache_read_tokens",
    "cache_creation_tokens",
    "total_prompt_tokens",
    "reasoning_tokens",
    "output_text_tokens",
    "total_completion_tokens",
    "cache_read_cost",
    "uncached_input_cost",
    "total_cost",
    "total_prompt_cost",
    "total_completion_cost",
]

# Shared latency/token/cost columns for the per-batch-size summary CSV.
# Prepend experiment-specific columns (accuracy, true/false positives, etc.)
# and a "batch_size" key to build the full SUMMARY_FIELDNAMES for each experiment.
SHARED_SUMMARY_FIELDNAMES = [
    "total_latency_secs",
    "total_llm_calls",
    "total_ttft_secs",
    "input_text_tokens",
    "total_cache_read_tokens",
    "cache_creation_tokens",
    "total_prompt_tokens",
    "reasoning_tokens",
    "output_text_tokens",
    "total_completion_tokens",
    "cache_read_cost",
    "uncached_input_cost",
    "total_cost",
    "total_prompt_cost",
    "total_completion_cost",
]


@dataclass
class BatchStats:
    model_name: str
    batch_size: int
    total_llm_calls: int
    total_docs_processed: int
    llm_call_duration_secs: float


class BatchedExecutor(ABC):
    """
    Template-method base for document-batching LLM executors.

    execute() runs the standard single-pass batch loop.
    Subclasses with multiple passes (e.g. one call per field) override execute()
    and call _run_batch_loop() once per pass.
    """

    # Subclasses override with their actual batch prompt.
    # Placeholders available: {num_docs}, {docs}
    BATCH_PROMPT_TEMPLATE: str = (
        "Process the following {num_docs} document(s):\n\n{docs}"
    )

    # Subclasses declare only their domain-specific extra columns.
    # The base class appends SHARED_PER_CALL_FIELDNAMES automatically.
    # Use per_call_all_fieldnames for the CSV header and DictWriter.
    #
    #   class MyExecutor(BatchedExecutor):
    #       PER_CALL_EXTRA_FIELDNAMES = ["num_docs_in_batch", "num_matches"]
    PER_CALL_EXTRA_FIELDNAMES: List[str] = []

    @property
    def per_call_all_fieldnames(self) -> List[str]:
        return self.PER_CALL_EXTRA_FIELDNAMES + SHARED_PER_CALL_FIELDNAMES

    _FORMAT_DESCRIPTIONS: Dict[str, str] = {
        "bullet": "Documents are provided in bullet-point format.",
        "json": "Documents are provided as a JSON array.",
        "paragraph": "Documents are provided as plain paragraphs.",
    }

    def __init__(
        self,
        model: Model,
        batch_size: int = 5,
        temperature: float = 0.0,
        prompt_order: str = "operator_first",
        doc_format: str = "bullet",
        reasoning_effort: str | None = None,
        include_thoughts: bool = False,
        disable_cache: bool = False,
        direct_provider: bool = False,
        use_vertex: bool = False,
        plain_json: bool = False,
        streaming: bool = False,
        use_local_vllm: bool = False,
        vllm_base_url: str = "http://localhost:8000/v1",
    ):
        assert prompt_order in ("operator_first", "documents_first")
        assert doc_format in self._FORMAT_DESCRIPTIONS

        self.model = model
        self.model_str = model.model_id
        self.batch_size = batch_size
        self.temperature = temperature
        self.prompt_order = prompt_order
        self.doc_format = doc_format
        self.reasoning_effort = reasoning_effort
        self.include_thoughts = include_thoughts
        self.disable_cache = disable_cache
        self.plain_json = plain_json
        self.streaming = streaming

        if use_local_vllm:
            self.client = VLLMClient(model, base_url=vllm_base_url)
        elif direct_provider:
            if model.is_provider_openai() or model.model_id.startswith("openai/"):
                self.client = OpenAIClient(model)
            else:
                self.client = GeminiClient(model, use_vertex=use_vertex)
        else:
            self.client = OpenRouterClient(model)

        # Aggregated usage stats
        self.aggregate_input_text_tokens = 0
        self.aggregate_total_cache_read_tokens = 0
        self.aggregate_cache_creation_tokens = 0
        self.aggregate_total_prompt_tokens = 0
        self.aggregate_reasoning_tokens = 0
        self.aggregate_output_text_tokens = 0
        self.aggregate_total_completion_tokens = 0
        self.aggregate_cache_read_cost = 0.0
        self.aggregate_uncached_input_cost = 0.0
        self.aggregate_total_cost = 0.0
        self.aggregate_prompt_cost = 0.0
        self.aggregate_completion_cost = 0.0
        self.aggregate_ttft_secs = 0.0

        self.run_id: int = 1
        self.per_call_csv_path = None
        self.per_call_json_path = None

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cache_buster(messages: list[dict]) -> list[dict]:
        nonce = f"[nonce:{uuid.uuid4().hex}] "
        result = []
        injected = False
        for msg in messages:
            if not injected and msg.get("role") == "system":
                msg = {**msg, "content": nonce + msg["content"]}
                injected = True
            result.append(msg)
        return result

    def _get_extra_call_kwargs(self, client_type: str) -> dict:
        """
        Extra kwargs forwarded to the client's generate() call.

        Override in subclasses that need structured output schemas
        (Pydantic text_format for OpenAI, response_json_schema for Gemini).
        """
        if client_type == "openai":
            return {
                "reasoning_effort": self.reasoning_effort,
                "include_thoughts": self.include_thoughts,
                "response_format": {"type": "json_object"},
            }
        if client_type == "gemini":
            return {
                "include_thoughts": self.include_thoughts,
                "response_mime_type": "application/json",
            }
        if client_type == "vllm":
            return {"response_format": {"type": "json_object"}}
        # openrouter
        return {
            "reasoning_effort": self.reasoning_effort,
            "response_format": {"type": "json_object"},
        }

    def _call_llm(
        self, messages: list[dict], max_retries: int = 5
    ) -> tuple[str, str, dict, float]:
        last_exception = None

        if isinstance(self.client, OpenAIClient):
            client_type = "openai"
        elif isinstance(self.client, GeminiClient):
            client_type = "gemini"
        elif isinstance(self.client, VLLMClient):
            client_type = "vllm"
        else:
            client_type = "openrouter"

        shared_kwargs = dict(messages=messages, temperature=self.temperature)
        if self.streaming:
            shared_kwargs["stream"] = True
        extra = self._get_extra_call_kwargs(client_type)

        for attempt in range(max_retries):
            try:
                start = time.time()
                response = self.client.generate(**shared_kwargs, **extra)
                latency = time.time() - start
                ttft = getattr(response, "ttft", None)
                return (
                    response.content,
                    getattr(response, "thought", ""),
                    response.usage,
                    latency,
                    ttft,
                )
            except Exception as e:
                last_exception = e
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s: %s",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries - 1:
                    delay = min(2**attempt + random.uniform(0, 1), 60)
                    time.sleep(delay)

        raise last_exception

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _update_aggregates(self, usage: dict) -> None:
        self.aggregate_input_text_tokens += usage.get("input_text_tokens", 0)
        self.aggregate_total_cache_read_tokens += usage.get(
            "total_cache_read_tokens", 0
        )
        self.aggregate_cache_creation_tokens += usage.get("cache_creation_tokens", 0)
        self.aggregate_total_prompt_tokens += usage.get("total_prompt_tokens", 0)
        self.aggregate_reasoning_tokens += usage.get("reasoning_tokens", 0)
        self.aggregate_output_text_tokens += usage.get("output_text_tokens", 0)
        self.aggregate_total_completion_tokens += usage.get(
            "total_completion_tokens", 0
        )
        self.aggregate_cache_read_cost += usage.get("cache_read_cost", 0.0)
        self.aggregate_uncached_input_cost += usage.get("uncached_input_cost", 0.0)
        self.aggregate_total_cost += usage.get("total_cost", 0.0)
        self.aggregate_prompt_cost += usage.get("total_prompt_cost", 0.0)
        self.aggregate_completion_cost += usage.get("total_completion_cost", 0.0)

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "input_text_tokens": self.aggregate_input_text_tokens,
            "total_cache_read_tokens": self.aggregate_total_cache_read_tokens,
            "cache_creation_tokens": self.aggregate_cache_creation_tokens,
            "total_prompt_tokens": self.aggregate_total_prompt_tokens,
            "reasoning_tokens": self.aggregate_reasoning_tokens,
            "output_text_tokens": self.aggregate_output_text_tokens,
            "total_completion_tokens": self.aggregate_total_completion_tokens,
            "cache_read_cost": self.aggregate_cache_read_cost,
            "uncached_input_cost": self.aggregate_uncached_input_cost,
            "total_cost": self.aggregate_total_cost,
            "total_prompt_cost": self.aggregate_prompt_cost,
            "total_completion_cost": self.aggregate_completion_cost,
            "total_ttft_secs": self.aggregate_ttft_secs,
        }

    @staticmethod
    def _make_shared_summary_csv_fields(batch_stats: "BatchStats", usage: dict) -> dict:
        """
        Build the shared latency/token/cost portion of a summary CSV row.

        Callers merge in their experiment-specific fields (accuracy, etc.):

            csv_row = {
                "batch_size": batch_size,
                **{k: metrics[k] for k in ["accuracy", "true_positives", ...]},
                **self._make_shared_summary_csv_fields(batch_stats, usage),
            }
        """
        return {
            "total_latency_secs": batch_stats.llm_call_duration_secs,
            "total_llm_calls": batch_stats.total_llm_calls,
            "total_ttft_secs": usage.get("total_ttft_secs", 0.0),
            **{
                k: usage.get(k, 0)
                for k in [
                    "input_text_tokens",
                    "total_cache_read_tokens",
                    "cache_creation_tokens",
                    "total_prompt_tokens",
                    "reasoning_tokens",
                    "output_text_tokens",
                    "total_completion_tokens",
                    "cache_read_cost",
                    "uncached_input_cost",
                    "total_cost",
                    "total_prompt_cost",
                    "total_completion_cost",
                ]
            },
        }

    @staticmethod
    def _make_shared_summary_json_fields(
        batch_stats: "BatchStats", usage: dict
    ) -> dict:
        """
        Build the shared latency/token/cost portion of a summary JSON entry.

        Callers merge in their experiment-specific fields:

            json_entry = {
                "batch_size": batch_size,
                "accuracy_metrics": metrics,
                **self._make_shared_summary_json_fields(batch_stats, usage),
                "example": {...},
            }
        """
        return {
            "total_latency_secs": batch_stats.llm_call_duration_secs,
            "total_llm_calls": batch_stats.total_llm_calls,
            "total_ttft_secs": usage.get("total_ttft_secs", 0.0),
            "input_token_details": {
                k: usage.get(k, 0)
                for k in [
                    "input_text_tokens",
                    "total_cache_read_tokens",
                    "cache_creation_tokens",
                    "total_prompt_tokens",
                ]
            },
            "output_token_details": {
                k: usage.get(k, 0)
                for k in [
                    "reasoning_tokens",
                    "output_text_tokens",
                    "total_completion_tokens",
                ]
            },
            "cost": {
                k: usage.get(k, 0.0)
                for k in [
                    "cache_read_cost",
                    "uncached_input_cost",
                    "total_prompt_cost",
                    "total_completion_cost",
                    "total_cost",
                ]
            },
        }

    def _make_shared_csv_row(
        self, batch_number: int, duration: float, usage: dict, ttft: float | None = None
    ) -> dict:
        """
        Build the shared portion of a per-call CSV row from SHARED_PER_CALL_FIELDNAMES.

        _process_batch implementations can call this and merge in their domain fields:

            row = self._make_shared_csv_row(state["batch_number"], duration, usage,
                                            ttft=state.get("ttft"))
            row["num_docs_in_batch"] = len(batch)
            self._write_per_call_csv(row)
        """
        return {
            "run_id": self.run_id,
            "batch_number": batch_number,
            "latency_secs": duration,
            "ttft_secs": ttft if ttft is not None else "",
            "input_text_tokens": usage.get("input_text_tokens", 0),
            "total_cache_read_tokens": usage.get("total_cache_read_tokens", 0),
            "cache_creation_tokens": usage.get("cache_creation_tokens", 0),
            "total_prompt_tokens": usage.get("total_prompt_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "output_text_tokens": usage.get("output_text_tokens", 0),
            "total_completion_tokens": usage.get("total_completion_tokens", 0),
            "cache_read_cost": usage.get("cache_read_cost", 0.0),
            "uncached_input_cost": usage.get("uncached_input_cost", 0.0),
            "total_cost": usage.get("total_cost", 0.0),
            "total_prompt_cost": usage.get("total_prompt_cost", 0.0),
            "total_completion_cost": usage.get("total_completion_cost", 0.0),
        }

    def _write_per_call_csv(self, row: dict) -> None:
        if not self.per_call_csv_path:
            return
        with open(self.per_call_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.per_call_all_fieldnames).writerow(row)

    @staticmethod
    def _make_shared_json_entry(
        batch_number: int, duration: float, usage: dict, thought: str, output: str,
        ttft: float | None = None,
    ) -> dict:
        """
        Build the shared portion of a per-call JSON entry.

        _process_batch implementations merge in domain-specific fields:

            entry = self._make_shared_json_entry(state["batch_number"], duration, usage, thought, output)
            entry["num_docs_in_batch"] = len(batch)
            entry["matches"] = [...]
            self._write_per_call_json(entry, state["batch_number"])
        """
        return {
            "batch_number": batch_number,
            "latency_secs": duration,
            "ttft_secs": ttft,
            "input_token_details": {
                "input_text_tokens": usage.get("input_text_tokens", 0),
                "total_cache_read_tokens": usage.get("total_cache_read_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_tokens", 0),
                "total_prompt_tokens": usage.get("total_prompt_tokens", 0),
            },
            "output_token_details": {
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "output_text_tokens": usage.get("output_text_tokens", 0),
                "total_completion_tokens": usage.get("total_completion_tokens", 0),
            },
            "cost": {
                "cache_read_cost": usage.get("cache_read_cost", 0.0),
                "uncached_input_cost": usage.get("uncached_input_cost", 0.0),
                "total_prompt_cost": usage.get("total_prompt_cost", 0.0),
                "total_completion_cost": usage.get("total_completion_cost", 0.0),
                "total_cost": usage.get("total_cost", 0.0),
            },
            "thought": thought,
            "response": output,
        }

    def _write_per_call_json(self, entry: dict, batch_number: int) -> None:
        if not self.per_call_json_path:
            return
        with open(self.per_call_json_path, "a", encoding="utf-8") as f:
            if batch_number > 1:
                f.write(",")
            f.write("\n  " + json.dumps(entry, indent=2).replace("\n", "\n  "))

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text

    @abstractmethod
    def _format_bullet(self, docs: List[Dict[str, Any]]) -> str: ...

    @abstractmethod
    def _format_json(self, docs: List[Dict[str, Any]]) -> str: ...

    @abstractmethod
    def _format_paragraph(self, docs: List[Dict[str, Any]]) -> str: ...

    def build_batch_prompt(self, docs: List[Dict[str, Any]]) -> str:
        formatters = {
            "bullet": self._format_bullet,
            "json": self._format_json,
            "paragraph": self._format_paragraph,
        }
        docs_text = formatters[self.doc_format](docs)
        return self.BATCH_PROMPT_TEMPLATE.format(num_docs=len(docs), docs=docs_text)

    def _build_messages(self, system_prompt: str, doc_prompt: str) -> list[dict]:
        if self.prompt_order == "documents_first":
            return [{"role": "user", "content": doc_prompt + "\n\n" + system_prompt}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": doc_prompt},
        ]

    # ------------------------------------------------------------------
    # Template-method hooks — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for a single-pass execution."""
        ...

    @abstractmethod
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        output: str,
        thought: str,
        usage: dict,
        duration: float,
        state: dict,
    ) -> dict | None:
        """
        Handle one batch's LLM response.

        Update state["results"] with parsed output. Call _write_per_call_csv()
        for the per-call row. Return a tqdm postfix dict or None for defaults.

        state keys guaranteed present:
          - results: list, initialized to []
          - batch_number: int, 1-indexed, already incremented before this call
        """
        ...

    def _finalize(self, state: dict, docs: List[Dict[str, Any]]) -> Any:
        """Return the final result from state. Override to reshape if needed."""
        return state["results"]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_batch_loop(
        self,
        docs: List[Dict[str, Any]],
        system_prompt: str,
        stats: BatchStats,
        state: dict,
        verbose: bool,
        desc: str = "Processing batches",
    ) -> None:
        """
        Run one full pass of the batch loop, mutating stats and state in place.

        Separate from execute() so multi-pass subclasses can call it once per pass
        (e.g. once per field in a separate-converts loop).
        """
        batches = [
            docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)
        ]
        pbar = tqdm(
            batches,
            desc=desc,
            disable=not verbose,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for batch in pbar:
            doc_prompt = self.build_batch_prompt(batch)
            messages = self._build_messages(system_prompt, doc_prompt)
            if self.disable_cache:
                messages = self._add_cache_buster(messages)

            try:
                output, thought, usage, duration, ttft = self._call_llm(messages)
            except Exception as e:
                logger.error("LLM call failed after retries: %s", e)
                continue

            stats.llm_call_duration_secs += duration
            stats.total_llm_calls += 1
            stats.total_docs_processed += len(batch)
            self._update_aggregates(usage)
            state["batch_number"] += 1
            state["messages"] = messages
            state["ttft"] = ttft
            if ttft is not None:
                self.aggregate_ttft_secs += ttft

            postfix = self._process_batch(
                batch, output, thought, usage, duration, state
            )
            pbar.set_postfix(
                postfix
                or {
                    "Cost": f"${self.aggregate_total_cost:.4f}",
                    "Cached": f"{self.aggregate_total_cache_read_tokens:,}",
                }
            )

        pbar.close()

    def execute(
        self, docs: List[Dict[str, Any]], verbose: bool = False
    ) -> tuple[Any, BatchStats]:
        """
        Standard single-pass execution: one system prompt, one loop over batches.

        Multi-pass tasks (e.g. a separate LLM call per field) should override this
        and call _run_batch_loop() once per pass with different system prompts.
        """
        print(f"Processing {len(docs)} documents with batch size {self.batch_size}")

        stats = BatchStats(
            model_name=self.model_str,
            batch_size=self.batch_size,
            total_llm_calls=0,
            total_docs_processed=0,
            llm_call_duration_secs=0.0,
        )
        state: dict = {"results": [], "batch_number": 0}

        if self.per_call_json_path:
            with open(self.per_call_json_path, "w", encoding="utf-8") as f:
                f.write("[")

        self._run_batch_loop(docs, self._get_system_prompt(), stats, state, verbose)

        if self.per_call_json_path:
            with open(self.per_call_json_path, "a", encoding="utf-8") as f:
                f.write("\n]")

        return self._finalize(state, docs), stats


# ---------------------------------------------------------------------------
# Summary file utilities
# ---------------------------------------------------------------------------


def init_summary_files(
    csv_path: Path,
    json_path: Path,
    fieldnames: List[str],
    settings: Dict[str, Any],
) -> bool:
    """
    Create or resume summary CSV and JSON output files.

    Returns True if the JSON results array is empty (new file or no entries yet),
    so the caller knows whether to prepend a comma before the first appended entry.
    """
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        print(f"Created CSV: {csv_path}")
    else:
        print(f"Appending to CSV: {csv_path}")

    if json_path.exists():
        with open(json_path, "r+", encoding="utf-8") as f:
            content = f.read().rstrip()
            if content.endswith("}"):
                content = content[:-1].rstrip()
            if content.endswith("]"):
                content = content[:-1].rstrip()
            marker = '"results": ['
            idx = content.rfind(marker)
            after_bracket = content[idx + len(marker) :].strip() if idx != -1 else ""
            json_first_entry = after_bracket == ""
            f.seek(0)
            f.truncate()
            f.write(content)
        print(f"Resuming JSON: {json_path}")
        return json_first_entry
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write('{\n  "settings": ')
            f.write(json.dumps(settings, indent=2).replace("\n", "\n  "))
            f.write(',\n  "results": [')
        print(f"Created JSON: {json_path}")
        return True
