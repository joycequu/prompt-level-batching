"""
TweetEval Irony Detection — single-file experiment using BatchedExecutor base class.

Query: Classify each tweet as ironic or non-ironic. Return tweetId of ironic tweets.

Data: data/tweeteval/irony_random_100.csv
  Columns: tweetId, text, label (0=non_irony, 1=irony), label_text
"""

import argparse
import csv
import json
import logging
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


class TweetResult(BaseModel):
    tweetId: str
    is_irony: bool


class BatchTweetResponse(BaseModel):
    results: List[TweetResult]


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
    "openai", "httpcore", "httpx", "urllib3", "asyncio",
    "google", "google.genai", "google.auth", "google.api_core",
]
for _name in _NOISY_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False


class TweetEvalIronyExecutor(BatchedExecutor):
    """Classify tweets as ironic or non-ironic, returning matching tweetIds."""

    BATCH_PROMPT_TEMPLATE = (
        "Evaluate the following {num_docs} tweet(s):\n\n{docs}\n\n"
        "For each tweet above, determine if it uses irony."
    )

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are a helpful assistant whose job is to detect irony in tweets.\n"
        "You will be presented with multiple tweets and need to determine if each tweet uses irony.\n\n"
        "{format_description}\n\n"
        "For each tweet, evaluate:\n"
        "- Does the tweet use irony (saying the opposite of what is meant, or using sarcasm)?\n\n"
        "Return your response as a JSON object with the following structure:\n"
        '{{"results": [{{"tweetId": "<id>", "is_irony": true or false}}, ...]}}'
    )

    _FORMAT_DESCRIPTIONS = {
        "bullet": "Tweets are provided in bullet-point format. Each tweet block starts with 'Tweet N:' and lists tweetId and text as bullet fields.",
        "json": "Tweets are provided as a JSON array. Each element is an object with tweetId and text fields.",
        "paragraph": "Tweets are provided as plain paragraphs. Each paragraph starts with the tweetId followed by a colon and the tweet text.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.example_request = None
        self.example_response = None

    @property
    def per_call_all_fieldnames(self) -> list:
        shared = SHARED_PER_CALL_FIELDNAMES
        return [shared[0], "num_tweets_in_batch"] + shared[1:] + ["num_irony"]

    def _get_extra_call_kwargs(self, client_type: str) -> dict:
        json_schema = BatchTweetResponse.model_json_schema()
        if client_type == "openai":
            return {
                "reasoning_effort": self.reasoning_effort,
                "include_thoughts": self.include_thoughts,
                "text_format": BatchTweetResponse,
            }
        if client_type == "gemini":
            return {
                "include_thoughts": self.include_thoughts,
                "response_mime_type": "application/json",
                "response_json_schema": json_schema,
            }
        # openrouter
        if self.plain_json:
            return {
                "reasoning_effort": self.reasoning_effort,
                "response_format": {"type": "json_object"},
            }
        return {
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "BatchTweetResponse",
                    "strict": True,
                    "schema": _strict_json_schema(json_schema),
                },
            },
        }

    def _get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            format_description=self._FORMAT_DESCRIPTIONS[self.doc_format]
        )

    def _format_bullet(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for i, t in enumerate(docs, 1):
            lines.append(f"Tweet {i}:")
            lines.append(f"- tweetId: {t['tweetId']}")
            lines.append(f"- text: {t['text']}")
        return "\n".join(lines)

    def _format_json(self, docs: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [{"tweetId": t["tweetId"], "text": t["text"]} for t in docs],
            indent=2,
        )

    def _format_paragraph(self, docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(f"{t['tweetId']}: {t['text']}" for t in docs)

    def _process_batch(self, batch, output, thought, usage, duration, state):
        count_before = len(state["results"])

        try:
            parsed = BatchTweetResponse.model_validate_json(
                self._strip_markdown_json(output)
            )
            for item in parsed.results:
                if item.is_irony:
                    state["results"].append(item.tweetId)
        except Exception as e:
            logger.error("Failed to parse response: %s\nOutput: %.200s", e, output)

        batch_irony_ids = state["results"][count_before:]
        batch_irony = len(batch_irony_ids)

        row = self._make_shared_csv_row(state["batch_number"], duration, usage, ttft=state.get("ttft"))
        row["num_tweets_in_batch"] = len(batch)
        row["num_irony"] = batch_irony
        self._write_per_call_csv(row)

        entry = self._make_shared_json_entry(
            state["batch_number"], duration, usage, thought, output, ttft=state.get("ttft")
        )
        entry["num_tweets_in_batch"] = len(batch)
        entry["tweet_ids"] = [t["tweetId"] for t in batch]
        entry["num_irony"] = batch_irony
        entry["irony_tweet_ids"] = batch_irony_ids
        self._write_per_call_json(entry, state["batch_number"])

        if self.example_request is None:
            self.example_request = state["messages"]
            self.example_response = output

        return {
            "Irony": len(state["results"]),
            "Cost": f"${self.aggregate_total_cost:.4f}",
            "Cached": f"{self.aggregate_total_cache_read_tokens:,}",
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tweets(filepath: str, num_tweets: int = None) -> List[Dict[str, Any]]:
    tweets = []
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tweets.append({"tweetId": row["tweetId"], "text": row["text"]})
            if num_tweets and len(tweets) >= num_tweets:
                break
    return tweets


def load_ground_truth(filepath: str) -> Dict[str, bool]:
    ground_truth = {}
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ground_truth[row["tweetId"]] = int(row["label"]) == 1
    return ground_truth


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def calculate_accuracy(
    irony_tweet_ids: List[str], ground_truth: Dict[str, bool]
) -> Dict[str, Any]:
    irony_set = set(irony_tweet_ids)
    tp = fp = tn = fn = 0
    for tweet_id, is_irony_gt in ground_truth.items():
        predicted = tweet_id in irony_set
        if is_irony_gt and predicted:
            tp += 1
        elif not is_irony_gt and not predicted:
            tn += 1
        elif not is_irony_gt and predicted:
            fp += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "total_tweets": total,
        "correct": tp + tn,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

SUMMARY_FIELDNAMES = [
    "batch_size",
    "run_id",
    "accuracy",
    "total_tweets",
    "correct",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
] + SHARED_SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TweetEval irony detection batch size experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, dest="output_dir_name")
    parser.add_argument(
        "--data",
        required=True,
        dest="data_file",
        help="Input CSV filename under data/tweeteval/ (contains both tweets and labels)",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 5, 10, 15, 20]
    )
    parser.add_argument("--num-tweets", type=int, default=None, dest="num_tweets")
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
        "--run-id", type=int, default=1, dest="run_id",
        help="Run identifier (default: 1). run_id > 1 appends to existing per-call CSVs instead of overwriting.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "results" / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).resolve().parent.parent / "data" / "tweeteval" / args.data_file

    print(f"Model: {args.model}")
    tweets = load_tweets(data_path, num_tweets=args.num_tweets)
    print(f"Loaded {len(tweets)} tweets")
    ground_truth = load_ground_truth(data_path)
    print(f"Loaded ground truth for {len(ground_truth)} tweets")

    settings = {
        "model": args.model,
        "direct_provider": args.direct_provider,
        "use_vertex": args.use_vertex,
        "include_thoughts": args.include_thoughts,
        "disable_cache": args.disable_cache,
        "reasoning_effort": args.reasoning_effort,
        "plain_json": args.plain_json,
        "streaming": args.streaming,
        "prompt_order": args.prompt_order,
        "doc_format": args.doc_format,
        "data_file": args.data_file,
        "num_tweets": args.num_tweets,
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

        executor = TweetEvalIronyExecutor(
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

        irony_ids, batch_stats = executor.execute(tweets, verbose=True)
        usage = executor.get_usage_summary()

        loaded_ids = {t["tweetId"] for t in tweets}
        scoped_gt = {k: v for k, v in ground_truth.items() if k in loaded_ids}
        metrics = calculate_accuracy(irony_ids, scoped_gt)

        num_irony = len(irony_ids)
        irony_rate = (
            num_irony / batch_stats.total_docs_processed
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
                    "total_tweets",
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
            "num_irony": num_irony,
            "irony_rate": irony_rate,
            "irony_tweet_ids": irony_ids,
            "total_tweets_processed": batch_stats.total_docs_processed,
            "accuracy_metrics": metrics,
            **executor._make_shared_summary_json_fields(batch_stats, usage),
            "example": {
                "request": executor.example_request or [],
                "response": executor.example_response or "",
            },
        }

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
