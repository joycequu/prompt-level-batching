"""
LitSearch — sem_filter experiment using BatchedExecutor base class.

Task: given a search query and a pool of candidate papers (title + abstract),
      return the paperIds of papers relevant to the query.

The LitSearch dataset has no pre-built candidate pool. We construct one by taking
all ground-truth papers for the query and padding with randomly sampled corpus
papers to reach --pool-size (default 100). The ground truth is always fully
included in the pool.

Metrics: precision, recall, F1.
Accuracy is omitted — it is inflated by the many true negatives in the sparse
retrieval setting (typically 1–2 positives out of 100 candidates).

Usage:
    python litsearch_adapted.py \\
        --model google/gemini-3-flash-preview \\
        --output-dir 01_or_q1_gem3flash_none_100 \\
        --query-index 0 \\
        --pool-size 100 \\
        --batch-sizes 1 2 5 10 20 50 100
"""

import argparse
import csv
import json
import logging
import random
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


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PaperResult(BaseModel):
    paperId: str
    is_relevant: bool


class BatchPaperResponse(BaseModel):
    results: List[PaperResult]


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


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class LitSearchFilterExecutor(BatchedExecutor):
    """Filter candidate papers by relevance to a search query, returning matching paperIds."""

    BATCH_PROMPT_TEMPLATE = (
        "Evaluate the following {num_docs} paper(s):\n\n{docs}\n\n"
        "For each paper above, determine if it is relevant to the search query."
    )

    _SYSTEM_PROMPT_TEMPLATE = (
        "You are a helpful assistant whose job is to identify papers relevant to a literature search query.\n"
        'Search query: "{query}"\n\n'
        "{format_description}\n\n"
        "For each paper, determine:\n"
        "- Is this paper relevant to the search query above?\n\n"
        "Return your response as a JSON object with the following structure:\n"
        '{{"results": [{{"paperId": "<id>", "is_relevant": true or false}}, ...]}}'
    )

    _FORMAT_DESCRIPTIONS = {
        "bullet": (
            "Papers are provided in bullet-point format. Each paper block starts with "
            "'Paper N:' and lists paperId, title, and abstract as bullet fields."
        ),
        "json": (
            "Papers are provided as a JSON array. Each element is an object with "
            "paperId, title, and abstract fields."
        ),
        "paragraph": (
            "Papers are provided as plain paragraphs. Each paragraph starts with the "
            "paperId followed by a colon, then the title and abstract."
        ),
    }

    def __init__(self, *args, query_text: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.query_text = query_text
        self.example_request = None
        self.example_response = None

    @property
    def per_call_all_fieldnames(self) -> list:
        shared = SHARED_PER_CALL_FIELDNAMES
        return [shared[0], "num_papers_in_batch"] + shared[1:] + ["num_relevant"]

    def _get_extra_call_kwargs(self, client_type: str) -> dict:
        json_schema = BatchPaperResponse.model_json_schema()
        if client_type == "openai":
            return {
                "reasoning_effort": self.reasoning_effort,
                "include_thoughts": self.include_thoughts,
                "text_format": BatchPaperResponse,
            }
        if client_type == "gemini":
            return {
                "include_thoughts": self.include_thoughts,
                "response_mime_type": "application/json",
                "response_json_schema": json_schema,
            }
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
                    "name": "BatchPaperResponse",
                    "strict": True,
                    "schema": _strict_json_schema(json_schema),
                },
            },
        }

    def _get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT_TEMPLATE.format(
            query=self.query_text,
            format_description=self._FORMAT_DESCRIPTIONS[self.doc_format],
        )

    def _format_bullet(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for i, p in enumerate(docs, 1):
            lines.append(f"Paper {i}:")
            lines.append(f"- paperId: {p['paperId']}")
            lines.append(f"- title: {p['title']}")
            lines.append(f"- abstract: {p['abstract']}")
        return "\n".join(lines)

    def _format_json(self, docs: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [
                {
                    "paperId": p["paperId"],
                    "title": p["title"],
                    "abstract": p["abstract"],
                }
                for p in docs
            ],
            indent=2,
        )

    def _format_paragraph(self, docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"{p['paperId']}: {p['title']}\n{p['abstract']}" for p in docs
        )

    def _process_batch(self, batch, output, thought, usage, duration, state):
        count_before = len(state["results"])

        try:
            parsed = BatchPaperResponse.model_validate_json(
                self._strip_markdown_json(output)
            )
            for item in parsed.results:
                if item.is_relevant:
                    state["results"].append(item.paperId)
        except Exception as e:
            logger.error("Failed to parse response: %s\nOutput: %.200s", e, output)

        batch_relevant_ids = state["results"][count_before:]
        batch_relevant = len(batch_relevant_ids)

        row = self._make_shared_csv_row(
            state["batch_number"], duration, usage, ttft=state.get("ttft")
        )
        row["num_papers_in_batch"] = len(batch)
        row["num_relevant"] = batch_relevant
        self._write_per_call_csv(row)

        entry = self._make_shared_json_entry(
            state["batch_number"],
            duration,
            usage,
            thought,
            output,
            ttft=state.get("ttft"),
        )
        entry["num_papers_in_batch"] = len(batch)
        entry["paper_ids"] = [p["paperId"] for p in batch]
        entry["num_relevant"] = batch_relevant
        entry["relevant_paper_ids"] = batch_relevant_ids
        self._write_per_call_json(entry, state["batch_number"])

        if self.example_request is None:
            self.example_request = state["messages"]
            self.example_response = output

        return {
            "Relevant": len(state["results"]),
            "Cost": f"${self.aggregate_total_cost:.4f}",
            "Cached": f"{self.aggregate_total_cache_read_tokens:,}",
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_corpus(corpus_data) -> Dict[str, Dict[str, str]]:
    """Build a {str(corpusid) -> {title, abstract}} lookup from corpus_clean."""
    return {
        str(row["corpusid"]): {"title": row["title"], "abstract": row["abstract"]}
        for row in corpus_data
    }


def load_query_entry(query_data, query_index: int) -> Dict[str, Any]:
    """Return the query record at query_index with string paper IDs."""
    row = query_data[query_index]
    return {
        "query_index": query_index,
        "query_set": row["query_set"],
        "query": row["query"],
        "specificity": row["specificity"],
        # corpusids are integers in the dataset; stringify for consistency
        "answer_doc_ids": [str(cid) for cid in row["corpusids"]],
    }


def build_candidate_pool(
    query_entry: Dict[str, Any],
    corpus: Dict[str, Dict[str, str]],
    pool_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build a candidate pool of exactly pool_size papers.

    All ground-truth papers are included first. The remainder are filled with
    randomly sampled papers from the corpus (excluding ground truth).
    Papers missing from corpus are skipped with a warning.
    """
    answer_ids = set(query_entry["answer_doc_ids"])

    pool = []
    missing_gt = []
    for pid in query_entry["answer_doc_ids"]:
        if pid in corpus:
            pool.append({"paperId": pid, **corpus[pid]})
        else:
            missing_gt.append(pid)
    if missing_gt:
        logger.warning(
            "%d ground-truth paper(s) not found in corpus and were skipped: %s",
            len(missing_gt), missing_gt,
        )

    n_needed = pool_size - len(pool)
    if n_needed > 0:
        candidates = [pid for pid in corpus if pid not in answer_ids]
        rng = random.Random(seed)
        sampled = rng.sample(candidates, min(n_needed, len(candidates)))
        for pid in sampled:
            pool.append({"paperId": pid, **corpus[pid]})

    # Shuffle so ground-truth papers aren't always at the front
    rng = random.Random(seed)
    rng.shuffle(pool)

    return pool


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def calculate_metrics(
    returned_ids: List[str],
    answer_doc_ids: List[str],
) -> Dict[str, Any]:
    """Compute precision, recall, and F1 against the ground-truth set."""
    returned_set = set(returned_ids)
    answer_set = set(answer_doc_ids)

    tp = len(returned_set & answer_set)
    fp = len(returned_set - answer_set)
    fn = len(answer_set - returned_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "num_ground_truth": len(answer_set),
    }


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

SUMMARY_FIELDNAMES = [
    "batch_size",
    "run_id",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "false_negatives",
    "num_ground_truth",
    "pool_size",
] + SHARED_SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LitSearch sem_filter experiments")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, dest="output_dir_name")
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        dest="query_index",
        help="0-based index into the LitSearch query split (597 queries total)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100,
        dest="pool_size",
        help="Total candidate pool size: ground-truth papers + random corpus samples",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50, 100]
    )
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
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming to record ttft_secs (time-to-first-token) in per-call CSV",
    )
    parser.add_argument(
        "--run-id", type=int, default=1, dest="run_id",
        help="Run identifier (default: 1). run_id > 1 appends to existing per-call CSVs instead of overwriting.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "results" / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LitSearch dataset...")
    from datasets import load_dataset

    query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
    corpus_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")

    corpus = load_corpus(corpus_data)
    print(f"Corpus size: {len(corpus):,} papers")

    query_entry = load_query_entry(query_data, args.query_index)
    print(
        f"Query [{args.query_index}] "
        f"({'broad' if query_entry['specificity'] == 0 else 'specific'}): "
        f"{query_entry['query']}"
    )
    print(
        f"Ground truth: {len(query_entry['answer_doc_ids'])} paper(s) — "
        f"IDs: {query_entry['answer_doc_ids']}"
    )

    papers = build_candidate_pool(query_entry, corpus, args.pool_size, args.seed)
    n_gt = len(query_entry["answer_doc_ids"])
    print(f"Candidate pool: {len(papers)} papers ({n_gt} relevant + {len(papers) - n_gt} random)")

    model = Model(args.model)

    settings = {
        "model": args.model,
        "query_index": args.query_index,
        "query": query_entry["query"],
        "query_set": query_entry["query_set"],
        "specificity": query_entry["specificity"],
        "answer_doc_ids": query_entry["answer_doc_ids"],
        "pool_size": len(papers),
        "num_ground_truth": len(query_entry["answer_doc_ids"]),
        "seed": args.seed,
        "direct_provider": args.direct_provider,
        "use_vertex": args.use_vertex,
        "include_thoughts": args.include_thoughts,
        "disable_cache": args.disable_cache,
        "reasoning_effort": args.reasoning_effort,
        "plain_json": args.plain_json,
        "streaming": args.streaming,
        "prompt_order": args.prompt_order,
        "doc_format": args.doc_format,
        "batch_sizes": args.batch_sizes,
    }

    csv_output = output_dir / "batch_size_summary.csv"
    json_output = output_dir / "batch_size_summary.json"
    json_first_entry = init_summary_files(
        csv_output, json_output, SUMMARY_FIELDNAMES, settings
    )

    for batch_size in args.batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"batch_size={batch_size}")
        print(f"{'=' * 80}")

        executor = LitSearchFilterExecutor(
            model,
            query_text=query_entry["query"],
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

        relevant_ids, batch_stats = executor.execute(papers, verbose=True)
        usage = executor.get_usage_summary()
        metrics = calculate_metrics(relevant_ids, query_entry["answer_doc_ids"])

        csv_row = {
            "batch_size": batch_size,
            "run_id": run_id,
            **{
                k: metrics[k]
                for k in [
                    "precision",
                    "recall",
                    "f1",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "num_ground_truth",
                ]
            },
            "pool_size": len(papers),
            **executor._make_shared_summary_csv_fields(batch_stats, usage),
        }

        json_entry = {
            "batch_size": batch_size,
            "run_id": run_id,
            "model_name": batch_stats.model_name,
            "query": query_entry["query"],
            "pool_size": len(papers),
            "num_relevant_returned": len(relevant_ids),
            "relevant_paper_ids": relevant_ids,
            "total_papers_processed": batch_stats.total_docs_processed,
            "metrics": metrics,
            **executor._make_shared_summary_json_fields(batch_stats, usage),
            "example": {
                "request": executor.example_request or [],
                "response": executor.example_response or "",
            },
        }

        print(
            f"\nPrecision: {metrics['precision']:.4f}  "
            f"Recall: {metrics['recall']:.4f}  "
            f"F1: {metrics['f1']:.4f}"
        )
        print(
            f"TP={metrics['true_positives']}  "
            f"FP={metrics['false_positives']}  "
            f"FN={metrics['false_negatives']}  "
            f"(ground truth: {metrics['num_ground_truth']} paper(s))"
        )
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
