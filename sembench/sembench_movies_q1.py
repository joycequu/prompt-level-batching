"""
SemBench Q1 — single-file experiment using BatchedExecutor base class.

Query: All positive reviews (any movie). Return reviewId.

This file replaces the two-file setup of:
  sembench_movies_q1_batch.py  (executor)
  run_sembench_batch_experiments.py  (runner)

SemBench experiments follow this pattern:
  1. Subclass BatchedExecutor — implement _get_system_prompt, _process_batch,
     and the three _format_* methods.
  2. Define load_data, load_ground_truth, evaluate functions.
  3. Write a main() that calls init_summary_files and loops over batch sizes.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models import Model
from utils.batch_executor_base import (
    BatchedExecutor,
    SHARED_PER_CALL_FIELDNAMES,
    SHARED_SUMMARY_FIELDNAMES,
    init_summary_files,
)


class ReviewResult(BaseModel):
    model_config = {"coerce_numbers_to_str": True}

    reviewId: str
    is_positive: bool


class BatchReviewResponse(BaseModel):
    results: List[ReviewResult]


def _strict_json_schema(schema: dict) -> dict:
    """Recursively add additionalProperties: false to every object in the schema."""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for key in ("properties", "$defs", "definitions"):
        if key in schema:
            schema[key] = {k: _strict_json_schema(v) for k, v in schema[key].items()}
    if "items" in schema:
        schema["items"] = _strict_json_schema(schema["items"])
    for key in ("anyOf", "allOf", "oneOf"):
        if key in schema:
            schema[key] = [_strict_json_schema(s) for s in schema[key]]
    return schema


logger = logging.getLogger(__name__)

_NOISY_LOGGERS = [
    "openai",
    "httpcore",
    "httpx",
    "urllib3",
    "asyncio",
    "google",
    "google.genai",
    "google.auth",
    "google.api_core",
]
for _name in _NOISY_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False


class SembenchQ1Executor(BatchedExecutor):
    """Filter movie reviews by positive sentiment, returning matching reviewIds."""

    BATCH_PROMPT_TEMPLATE = (
        "Evaluate the following {num_docs} review(s):\n\n{docs}\n\n"
        "For each review above, determine if it is clearly positive."
    )

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are a helpful assistant whose job is to filter movie reviews based on sentiment.\n"
        "You will be presented with multiple reviews and need to determine if each review is clearly positive.\n\n"
        "{format_description}\n\n"
        "For each review, evaluate:\n"
        "- Is the review clearly positive in sentiment?\n\n"
        "Return your response as a JSON object with the following structure:\n"
        '{{"results": [{{"reviewId": "<id>", "is_positive": true or false}}, ...]}}'
    )

    _SYSTEM_PROMPT_TEMPLATE_COMPACT = (
        "You are a helpful assistant whose job is to filter movie reviews based on sentiment.\n"
        "You will be presented with multiple reviews and need to determine if each review is clearly positive.\n\n"
        "{format_description}\n\n"
        "For each review, evaluate:\n"
        "- Is the review clearly positive in sentiment?\n\n"
        "Return your response as a JSON object with the following structure:\n"
        '{{"results": [["<id>", true or false], ...]}}'
    )

    _COMPACT_SCHEMA = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "boolean"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }

    _FORMAT_DESCRIPTIONS = {
        "bullet": "Reviews are provided in bullet-point format. Each review block starts with 'Review N:' and lists reviewId and reviewText as bullet fields.",
        "json": "Reviews are provided as a JSON array. Each element is an object with reviewId and reviewText fields.",
        "paragraph": "Reviews are provided as plain paragraphs. Each paragraph starts with the reviewId followed by a colon and the review text.",
    }

    def __init__(self, *args, compact: bool = False, max_output_tokens: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact = compact
        self.max_output_tokens = max_output_tokens
        self.example_request = None
        self.example_response = None

    @property
    def per_call_all_fieldnames(self) -> list:
        shared = SHARED_PER_CALL_FIELDNAMES
        return [shared[0], "num_reviews_in_batch"] + shared[1:] + ["num_positive"]

    def _get_extra_call_kwargs(self, client_type: str) -> dict:
        if self.compact:
            schema = self._COMPACT_SCHEMA
            if client_type == "openai":
                return {
                    "reasoning_effort": self.reasoning_effort,
                    "include_thoughts": self.include_thoughts,
                    "response_format": {"type": "json_schema", "json_schema": {"name": "CompactReviewResponse", "strict": True, "schema": schema}},
                }
            if client_type == "gemini":
                return {
                    "include_thoughts": self.include_thoughts,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                }
            if client_type == "vllm":
                kw = {"response_format": {"type": "json_object"}}
                if self.max_output_tokens is not None:
                    kw["max_tokens"] = self.max_output_tokens
                return kw
            # openrouter
            extra = {"max_tokens": self.max_output_tokens} if self.max_output_tokens is not None else {}
            if self.plain_json:
                return {"reasoning_effort": self.reasoning_effort, "response_format": {"type": "json_object"}, **extra}
            return {
                "reasoning_effort": self.reasoning_effort,
                "response_format": {"type": "json_schema", "json_schema": {"name": "CompactReviewResponse", "strict": True, "schema": schema}},
                **extra,
            }

        json_schema = BatchReviewResponse.model_json_schema()
        if client_type == "openai":
            return {
                "reasoning_effort": self.reasoning_effort,
                "include_thoughts": self.include_thoughts,
                "text_format": BatchReviewResponse,
            }
        if client_type == "gemini":
            return {
                "include_thoughts": self.include_thoughts,
                "response_mime_type": "application/json",
                "response_json_schema": json_schema,
            }
        if client_type == "vllm":
            kw = {"response_format": {"type": "json_object"}}
            if self.max_output_tokens is not None:
                kw["max_tokens"] = self.max_output_tokens
            return kw
        # openrouter
        extra = {"max_tokens": self.max_output_tokens} if self.max_output_tokens is not None else {}
        if self.plain_json:
            return {
                "reasoning_effort": self.reasoning_effort,
                "response_format": {"type": "json_object"},
                **extra,
            }
        return {
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "BatchReviewResponse",
                    "strict": True,
                    "schema": _strict_json_schema(json_schema),
                },
            },
            **extra,
        }

    def _get_system_prompt(self) -> str:
        template = self._SYSTEM_PROMPT_TEMPLATE_COMPACT if self.compact else self._SYSTEM_PROMPT_TEMPLATE
        return template.format(
            format_description=self._FORMAT_DESCRIPTIONS[self.doc_format]
        )

    def _format_bullet(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for i, r in enumerate(docs, 1):
            lines.append(f"Review {i}:")
            lines.append(f"- reviewId: {r['reviewId']}")
            lines.append(f"- reviewText: {r['reviewText']}")
        return "\n".join(lines)

    def _format_json(self, docs: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [{"reviewId": r["reviewId"], "reviewText": r["reviewText"]} for r in docs],
            indent=2,
        )

    def _format_paragraph(self, docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(f"{r['reviewId']}: {r['reviewText']}" for r in docs)

    def _process_batch(self, batch, output, thought, usage, duration, state):
        count_before = len(state["results"])

        try:
            if self.compact:
                try:
                    data = json.loads(self._strip_markdown_json(output))
                    entries = data["results"]
                except Exception:
                    # Output was truncated — recover complete tuples via regex
                    entries = [
                        (m.group(1), m.group(2) == "true")
                        for m in re.finditer(r'\["(\d+)",\s*(true|false)\]', output)
                    ]
                    logger.warning("Truncated output; recovered %d entries via regex", len(entries))
                for entry in entries:
                    review_id, is_positive = str(entry[0]), bool(entry[1])
                    if is_positive:
                        state["results"].append(review_id)
            else:
                parsed = BatchReviewResponse.model_validate_json(
                    self._strip_markdown_json(output)
                )
                for item in parsed.results:
                    if item.is_positive:
                        state["results"].append(item.reviewId)
        except Exception as e:
            logger.error("Failed to parse response: %s\nOutput: %.200s", e, output)

        batch_positive_ids = state["results"][count_before:]
        batch_positive = len(batch_positive_ids)

        row = self._make_shared_csv_row(state["batch_number"], duration, usage, ttft=state.get("ttft"))
        row["num_reviews_in_batch"] = len(batch)
        row["num_positive"] = batch_positive
        self._write_per_call_csv(row)

        entry = self._make_shared_json_entry(
            state["batch_number"], duration, usage, thought, output, ttft=state.get("ttft")
        )
        entry["num_reviews_in_batch"] = len(batch)
        entry["review_ids"] = [r["reviewId"] for r in batch]
        entry["num_positive"] = batch_positive
        entry["positive_review_ids"] = batch_positive_ids
        self._write_per_call_json(entry, state["batch_number"])

        if self.example_request is None:
            self.example_request = state["messages"]
            self.example_response = output

        return {
            "Positive": len(state["results"]),
            "Cost": f"${self.aggregate_total_cost:.4f}",
            "Cached": f"{self.aggregate_total_cache_read_tokens:,}",
        }


def load_reviews(filepath: str, num_reviews: int = None) -> List[Dict[str, Any]]:
    reviews = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews.append(
                {
                    "id": row.get("id", ""),
                    "reviewId": row.get("reviewId", ""),
                    "reviewText": row.get("reviewText", ""),
                }
            )
            if num_reviews and len(reviews) >= num_reviews:
                break
    return reviews


def load_ground_truth(filepath: str) -> Dict[str, bool]:
    ground_truth = {}
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ground_truth[row["reviewId"]] = (
                row.get("scoreSentiment", "").lower() == "positive"
            )
    return ground_truth


# Evaluation
def calculate_accuracy(
    positive_review_ids: List[str], ground_truth: Dict[str, bool]
) -> Dict[str, Any]:
    positive_set = set(positive_review_ids)
    tp = fp = tn = fn = 0
    for review_id, is_pos_gt in ground_truth.items():
        predicted = review_id in positive_set
        if is_pos_gt and predicted:
            tp += 1
        elif not is_pos_gt and not predicted:
            tn += 1
        elif not is_pos_gt and predicted:
            fp += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "total_reviews": total,
        "correct": tp + tn,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


# Summary schema
SUMMARY_FIELDNAMES = [
    "batch_size",
    "run_id",
    "accuracy",
    "total_reviews",
    "correct",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
] + SHARED_SUMMARY_FIELDNAMES


# CLI
def main():
    parser = argparse.ArgumentParser(description="SemBench Q1 batch size experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, dest="output_dir_name")
    parser.add_argument(
        "--data",
        required=True,
        dest="data_file",
        help="Input CSV filename under data/sembench_movies/",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        dest="ground_truth_file",
        help="Ground truth CSV filename under data/sembench_movies/",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 5, 10, 15, 20]
    )
    parser.add_argument("--num-reviews", type=int, default=None, dest="num_reviews")
    parser.add_argument("--direct-provider", action="store_true", default=False)
    parser.add_argument("--use-vertex", action="store_true", default=False)
    parser.add_argument("--include-thoughts", action="store_true", default=False)
    parser.add_argument("--disable-cache", action="store_true", default=False)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--plain-json", action="store_true", default=False)
    parser.add_argument(
        "--prompt-order",
        default="operator_first",
        choices=["operator_first", "documents_first"],
    )
    parser.add_argument(
        "--doc-format", default="bullet", choices=["bullet", "json", "paragraph"]
    )
    parser.add_argument(
        "--streaming", action="store_true", default=False,
        help="Use streaming to record ttft_secs (time-to-first-token) in per-call CSV",
    )
    parser.add_argument(
        "--local", action="store_true", default=False,
        help="Use a local vLLM server instead of OpenRouter",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1", dest="vllm_url",
        help="Base URL of the vLLM server (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--compact", action="store_true", default=False,
        help="Use compact tuple output [[id, bool], ...] instead of named JSON objects to reduce output tokens",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=None, dest="max_output_tokens",
        help="Cap output tokens per call. When set to 1, accuracy metrics are skipped.",
    )
    parser.add_argument(
        "--num-repeats", type=int, default=None, dest="num_repeats",
        help="For each batch size, send exactly this many requests (uses num_repeats * batch_size reviews). Falls back to all loaded reviews if not enough.",
    )
    parser.add_argument(
        "--run-id", type=int, default=1, dest="run_id",
        help="Run identifier (default: 1). run_id > 1 appends to existing per-call CSVs instead of overwriting.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "results" / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = f"data/sembench_movies/{args.data_file}"
    ground_truth_path = f"data/sembench_movies/{args.ground_truth_file}"

    print(f"Model: {args.model}")
    reviews = load_reviews(data_path, num_reviews=args.num_reviews)
    print(f"Loaded {len(reviews)} reviews")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded ground truth for {len(ground_truth)} reviews")

    settings = {
        "model": args.model,
        "direct_provider": args.direct_provider,
        "use_vertex": args.use_vertex,
        "include_thoughts": args.include_thoughts,
        "disable_cache": args.disable_cache,
        "reasoning_effort": args.reasoning_effort,
        "plain_json": args.plain_json,
        "streaming": args.streaming,
        "local": args.local,
        "vllm_url": args.vllm_url,
        "compact": args.compact,
        "max_output_tokens": args.max_output_tokens,
        "num_repeats": args.num_repeats,
        "prompt_order": args.prompt_order,
        "doc_format": args.doc_format,
        "data_file": args.data_file,
        "ground_truth_file": args.ground_truth_file,
        "num_reviews": args.num_reviews,
        "batch_sizes": args.batch_sizes,
    }

    csv_output = output_dir / "batch_size_summary.csv"
    json_output = output_dir / "batch_size_summary.json"
    json_first_entry = init_summary_files(
        csv_output, json_output, SUMMARY_FIELDNAMES, settings
    )

    model = Model(args.model)

    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"batch_size={batch_size}")
        print(f"{'=' * 80}")

        executor = SembenchQ1Executor(
            model,
            batch_size=batch_size,
            direct_provider=args.direct_provider,
            use_vertex=args.use_vertex,
            include_thoughts=args.include_thoughts,
            disable_cache=args.disable_cache,
            reasoning_effort=args.reasoning_effort,
            plain_json=args.plain_json,
            streaming=args.streaming,
            prompt_order=args.prompt_order,
            doc_format=args.doc_format,
            use_local_vllm=args.local,
            vllm_base_url=args.vllm_url,
            compact=args.compact,
            max_output_tokens=args.max_output_tokens,
        )

        run_id = args.run_id
        executor.run_id = run_id
        per_call_csv = output_dir / f"batch_size_{batch_size}_per_call.csv"
        if run_id == 1 or not per_call_csv.exists():
            with open(per_call_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=executor.per_call_all_fieldnames).writeheader()
        executor.per_call_csv_path = per_call_csv
        json_stem = (
            f"batch_size_{batch_size}_per_call"
            if run_id == 1
            else f"batch_size_{batch_size}_run_{run_id}_per_call"
        )
        executor.per_call_json_path = output_dir / f"{json_stem}.json"

        positive_ids, batch_stats = executor.execute(reviews, verbose=True)
        usage = executor.get_usage_summary()

        skip_accuracy = args.max_output_tokens is not None
        if skip_accuracy:
            metrics = {
                "accuracy": None,
                "total_reviews": None,
                "correct": None,
                "true_positives": None,
                "false_positives": None,
                "true_negatives": None,
                "false_negatives": None,
            }
            num_positive = None
            positive_rate = None
        else:
            loaded_ids = {r["reviewId"] for r in reviews}
            scoped_gt = {k: v for k, v in ground_truth.items() if k in loaded_ids}
            metrics = calculate_accuracy(positive_ids, scoped_gt)
            num_positive = len(positive_ids)
            positive_rate = (
                num_positive / batch_stats.total_docs_processed
                if batch_stats.total_docs_processed > 0
                else 0.0
            )

        csv_row = {
            "batch_size": batch_size,
            "run_id": run_id,
            **{
                k: metrics[k]
                for k in [
                    "accuracy",
                    "total_reviews",
                    "correct",
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                ]
            },
            **executor._make_shared_summary_csv_fields(batch_stats, usage),
        }

        json_entry = {
            "batch_size": batch_size,
            "run_id": run_id,
            "model_name": batch_stats.model_name,
            "num_positive": num_positive,
            "positive_rate": positive_rate,
            "positive_review_ids": positive_ids if not skip_accuracy else [],
            "total_reviews_processed": batch_stats.total_docs_processed,
            "accuracy_metrics": metrics,
            **executor._make_shared_summary_json_fields(batch_stats, usage),
            "example": {
                "request": executor.example_request or [],
                "response": executor.example_response or "",
            },
        }

        if not skip_accuracy:
            print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print(f"Total cost: ${usage['total_cost']:.4f}")
        print(f"Total latency: {batch_stats.llm_call_duration_secs:.2f}s")

        with open(csv_output, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES).writerow(csv_row)

        with open(json_output, "a", encoding="utf-8") as f:
            if not json_first_entry:
                f.write(",")
            f.write("\n  " + json.dumps(json_entry, indent=2).replace("\n", "\n  "))
        json_first_entry = False

    with open(json_output, "a", encoding="utf-8") as f:
        f.write("\n  ]\n}")

    print(f"\n{'=' * 80}")
    print(f"Summary CSV: {csv_output}")
    print(f"Summary JSON: {json_output}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
