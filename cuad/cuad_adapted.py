"""
CUAD Contract Analysis — single-file experiment using BatchedExecutor base class.

Replaces the two-file setup of:
  cuad_demo_adapted.py        (executor)
  run_cuad_batch_experiments.py  (runner)

fields_per_call controls how many of the 41 CUAD fields are extracted per LLM call:
  --fields-per-call 41   one call per document batch  (formerly "one-convert")
  --fields-per-call 1    one call per field per batch (formerly "separate-converts")
  --fields-per-call K    ceil(41/K) calls per batch   ("field-chunking")

Script adapted from: https://github.com/mitdbg/palimpzest/blob/main/abacus-research/cuad-demo.py
"""

import argparse
import csv
import json
import logging
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models import Model
from utils.batch_executor_base import (
    BatchedExecutor,
    BatchStats,
    SHARED_PER_CALL_FIELDNAMES,
    SHARED_SUMMARY_FIELDNAMES,
    init_summary_files,
)
from cuad_data_loader import load_cuad_data, sample_contracts

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


CUAD_CATEGORIES = [
    {
        "Category": "Document Name",
        "Description": "The name of the contract",
        "Answer Format": "Contract Name",
    },
    {
        "Category": "Parties",
        "Description": "The two or more parties who signed the contract",
        "Answer Format": "Entity or individual names",
    },
    {
        "Category": "Agreement Date",
        "Description": "The date of the contract",
        "Answer Format": "Date (mm/dd/yyyy)",
    },
    {
        "Category": "Effective Date",
        "Description": "The date when the contract is effective",
        "Answer Format": "Date (mm/dd/yyyy)",
    },
    {
        "Category": "Expiration Date",
        "Description": "On what date will the contract's initial term expire?",
        "Answer Format": "Date (mm/dd/yyyy) / Perpetual",
    },
    {
        "Category": "Renewal Term",
        "Description": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
        "Answer Format": "[Successive] number of years/months / Perpetual",
    },
    {
        "Category": "Notice Period to Terminate Renewal",
        "Description": "What is the notice period required to terminate renewal?",
        "Answer Format": "Number of days/months/year(s)",
    },
    {
        "Category": "Governing Law",
        "Description": "Which state/country's law governs the interpretation of the contract?",
        "Answer Format": "Name of a US State / non-US Province, Country",
    },
    {
        "Category": "Most Favored Nation",
        "Description": "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Non-Compete",
        "Description": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Exclusivity",
        "Description": "Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all requirements from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "No-Solicit of Customers",
        "Description": "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Competitive Restriction Exception",
        "Description": "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "No-Solicit of Employees",
        "Description": "Is there a restriction on a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Non-Disparagement",
        "Description": "Is there a requirement on a party not to disparage the counterparty?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Termination for Convenience",
        "Description": "Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Rofr/Rofo/Rofn",
        "Description": "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Change of Control",
        "Description": "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Anti-Assignment",
        "Description": "Is consent or notice required of a party if the contract is assigned to a third party?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Revenue/Profit Sharing",
        "Description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Price Restrictions",
        "Description": "Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Minimum Commitment",
        "Description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Volume Restriction",
        "Description": "Is there a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "IP Ownership Assignment",
        "Description": "Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Joint IP Ownership",
        "Description": "Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "License Grant",
        "Description": "Does the contract contain a license granted by one party to its counterparty?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Non-Transferable License",
        "Description": "Does the contract limit the ability of a party to transfer the license being granted to a third party?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Affiliate License-Licensor",
        "Description": "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Affiliate License-Licensee",
        "Description": "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Unlimited/All-You-Can-Eat-License",
        "Description": "Is there a clause granting one party an 'enterprise,' 'all you can eat' or unlimited usage license?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Irrevocable or Perpetual License",
        "Description": "Does the contract contain a license grant that is irrevocable or perpetual?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Source Code Escrow",
        "Description": "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Post-Termination Services",
        "Description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Audit Rights",
        "Description": "Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Uncapped Liability",
        "Description": "Is a party's liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Cap on Liability",
        "Description": "Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Liquidated Damages",
        "Description": "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Warranty Duration",
        "Description": "What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?",
        "Answer Format": "Number of months or years",
    },
    {
        "Category": "Insurance",
        "Description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Covenant Not to Sue",
        "Description": "Is a party restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?",
        "Answer Format": "Yes/No",
    },
    {
        "Category": "Third Party Beneficiary",
        "Description": "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?",
        "Answer Format": "Yes/No",
    },
]

_N_FIELDS = len(CUAD_CATEGORIES)  # 41


class CUADExecutor(BatchedExecutor):
    """Extract CUAD fields from legal contracts.

    fields_per_call controls how many of the 41 fields are extracted per LLM call.
    """

    BATCH_PROMPT_TEMPLATE = (
        "Analyze the following {num_docs} contract(s):\n\n{docs}\n\n"
        "Extract the required information from each contract above."
    )

    _FORMAT_DESCRIPTIONS = {
        "bullet": "Contracts are provided in bullet-point format. Each contract block starts with 'Contract N:' and lists contract_id, title, and contract text as bullet fields.",
        "json": "Contracts are provided as a JSON array. Each element is an object with contract_id, title, and contract fields.",
        "paragraph": "Contracts are provided as plain paragraphs. Each paragraph starts with the contract_id followed by a colon, then the title and contract text.",
    }

    def __init__(self, *args, fields_per_call: int = 41, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            1 <= fields_per_call <= _N_FIELDS
        ), f"fields_per_call must be 1–{_N_FIELDS}"
        self.fields_per_call = fields_per_call

    @staticmethod
    def _build_response_schema(fields: List[Dict]) -> dict:
        field_properties = {"contract_id": {"type": "string"}}
        for cat in fields:
            field_properties[cat["Category"]] = {
                "type": "array",
                "items": {"type": "string"},
            }
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": field_properties,
                        "required": list(field_properties.keys()),
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }

    def _get_extra_call_kwargs(self, client_type: str) -> dict:
        fields = getattr(self, "_current_fields", CUAD_CATEGORIES)
        if client_type == "openrouter":
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
                        "name": "cuad_results",
                        "strict": True,
                        "schema": self._build_response_schema(fields),
                    },
                },
            }
        # Gemini direct and OpenAI fall back to base class behaviour
        return super()._get_extra_call_kwargs(client_type)

    @property
    def per_call_all_fieldnames(self) -> list:
        shared = SHARED_PER_CALL_FIELDNAMES
        return [
            shared[0],
            "field_chunk",
            "num_contracts_in_batch",
            "contract_ids",
        ] + shared[1:]

    def _build_system_prompt(self, fields: List[Dict]) -> str:
        field_descriptions = "\n".join(
            f"- {cat['Category']}: {cat['Description']}" for cat in fields
        )
        field_json_keys = ", ".join(f'"{cat["Category"]}": [...]' for cat in fields)
        n = len(fields)
        return (
            "You are a helpful assistant whose job is to extract structured information from legal contracts.\n"
            f"You will be presented with one or more contracts and need to extract the following {n} field(s) from each contract.\n\n"
            "Extract these fields (quote text spans verbatim, do not summarize or paraphrase):\n"
            f"{field_descriptions}\n\n"
            "Return your response as a JSON object with the following structure:\n"
            f'{{"results": [{{"contract_id": "<id>", {field_json_keys}}}, ...]}}\n\n'
            "Each field should contain a list of text spans (strings) extracted from the contract. "
            "If no spans exist for a field, return an empty list []."
        )

    def _get_system_prompt(self) -> str:
        return self._build_system_prompt(CUAD_CATEGORIES)

    def _format_bullet(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for i, c in enumerate(docs, 1):
            lines.append(f"Contract {i}:")
            lines.append(f"- contract_id: {c['contract_id']}")
            lines.append(f"- title: {c['title']}")
            lines.append(f"- contract text: {c['contract']}")
        return "\n".join(lines)

    def _format_json(self, docs: List[Dict[str, Any]]) -> str:
        return json.dumps(
            [
                {
                    "contract_id": c["contract_id"],
                    "title": c["title"],
                    "contract": c["contract"],
                }
                for c in docs
            ],
            indent=2,
        )

    def _format_paragraph(self, docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"{c['contract_id']}: {c['title']}\n{c['contract']}" for c in docs
        )

    def _process_batch(self, batch, output, thought, usage, duration, state):
        current_fields = state.get("current_fields", CUAD_CATEGORIES)
        field_names = [c["Category"] for c in current_fields]
        field_chunk_label = state.get("field_chunk_label", "all_fields")

        batch_results = []
        try:
            cleaned = self._strip_markdown_json(output)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                from json_repair import repair_json
                parsed = json.loads(repair_json(cleaned))
                logger.warning("Used json_repair for batch %d", state["batch_number"])
            if "results" in parsed:
                batch_results = parsed["results"]
                for result in batch_results:
                    contract_id = result.get("contract_id")
                    for i, r in enumerate(state["results"]):
                        if r["contract_id"] == contract_id:
                            for field_name in field_names:
                                state["results"][i][field_name] = result.get(
                                    field_name, []
                                )
                            break
        except Exception as e:
            logger.error("Failed to parse JSON: %s\nOutput: %.200s", e, output)

        row = self._make_shared_csv_row(state["batch_number"], duration, usage, ttft=state.get("ttft"))
        row["field_chunk"] = field_chunk_label
        row["num_contracts_in_batch"] = len(batch)
        row["contract_ids"] = "|".join(c["contract_id"] for c in batch)
        self._write_per_call_csv(row)

        entry = self._make_shared_json_entry(
            state["batch_number"], duration, usage, thought, output, ttft=state.get("ttft")
        )
        entry["field_chunk"] = field_chunk_label
        entry["num_contracts_in_batch"] = len(batch)
        entry["contract_ids"] = [c["contract_id"] for c in batch]
        entry["results"] = batch_results
        self._write_per_call_json(entry, state["batch_number"])

        return {
            "Cost": f"${self.aggregate_total_cost:.4f}",
            "Cached": f"{self.aggregate_total_cache_read_tokens:,}",
        }

    def execute(self, docs: List[Dict[str, Any]], verbose: bool = False):
        stats = BatchStats(
            model_name=self.model_str,
            batch_size=self.batch_size,
            total_llm_calls=0,
            total_docs_processed=0,
            llm_call_duration_secs=0.0,
        )

        if self.per_call_json_path:
            with open(self.per_call_json_path, "w", encoding="utf-8") as f:
                f.write("[")

        K = self.fields_per_call
        field_chunks = [CUAD_CATEGORIES[i : i + K] for i in range(0, _N_FIELDS, K)]
        n_chunks = len(field_chunks)
        n_doc_batches = -(-len(docs) // self.batch_size)  # ceil division

        print(f"Processing {len(docs)} contracts with batch size {self.batch_size}")
        print(
            f"Fields per call: {K}  |  Field chunks: {n_chunks}  |  Doc batches: {n_doc_batches}  |  Total calls: {n_chunks * n_doc_batches}"
        )

        state: dict = {
            "results": [{"contract_id": c["contract_id"]} for c in docs],
            "batch_number": 0,
            "current_fields": None,
            "field_chunk_label": None,
        }

        for chunk_idx, chunk in enumerate(field_chunks):
            first_field = chunk[0]["Category"]
            if n_chunks == 1:
                label = "all_fields"
            elif K == 1:
                label = first_field
            else:
                label = f"chunk{chunk_idx + 1}of{n_chunks}:{first_field}"

            state["current_fields"] = chunk
            state["field_chunk_label"] = label
            self._current_fields = chunk

            system_prompt = self._build_system_prompt(chunk)
            desc = (
                first_field
                if K == 1
                else f"Chunk {chunk_idx + 1}/{n_chunks}: {first_field}..."
            )
            self._run_batch_loop(docs, system_prompt, stats, state, verbose, desc=desc)

        # _run_batch_loop increments total_docs_processed per doc per chunk; correct it
        stats.total_docs_processed = len(docs)

        if self.per_call_json_path:
            with open(self.per_call_json_path, "a", encoding="utf-8") as f:
                f.write("\n]")

        return self._finalize(state, docs), stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_contracts(
    num_contracts: int, split: str = "test", seed: int = 42
) -> List[Dict[str, Any]]:
    dataset = load_cuad_data(split=split)
    filtered_dataset, sampled_titles = sample_contracts(dataset, num_contracts, seed)
    contracts = []
    for title in sampled_titles:
        contract_rows = [row for row in filtered_dataset if row["title"] == title]
        if contract_rows:
            contracts.append(
                {
                    "contract_id": contract_rows[0]["id"],
                    "title": title,
                    "contract": contract_rows[0]["context"],
                }
            )
    return contracts


def load_ground_truth(
    num_contracts: int, split: str = "test", seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    dataset = load_cuad_data(split=split)
    filtered_dataset, sampled_titles = sample_contracts(dataset, num_contracts, seed)
    category_names = [cat["Category"] for cat in CUAD_CATEGORIES]

    ground_truth = {}
    for title in sampled_titles:
        contract_rows = [row for row in filtered_dataset if row["title"] == title]
        if not contract_rows:
            continue
        contract_id = contract_rows[0]["id"]
        labels = {category: [] for category in category_names}
        for row in contract_rows:
            category_name = row["id"].split("__")[-1].split("_")[0].strip()
            category_name = category_name.replace(" For ", " for ")
            category_name = category_name.replace(" Of ", " of ")
            category_name = category_name.replace(" On ", " on ")
            category_name = category_name.replace(" Or ", " or ")
            category_name = category_name.replace(" To ", " to ")
            category_name = category_name.replace("Ip", "IP")
            if category_name not in category_names:
                print(f"Warning: Unknown category {category_name}")
                continue
            if isinstance(row["answers"], list):
                answer_texts = (
                    [ans["text"] for ans in row["answers"]] if row["answers"] else []
                )
            else:
                answer_texts = row["answers"].get("text", [])
            labels[category_name].extend(answer_texts)
        ground_truth[unicodedata.normalize("NFC", contract_id)] = labels
    return ground_truth


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

IOU_THRESH = 0.15


def get_jaccard(label: str, pred: str) -> float:
    remove_tokens = [c for c in string.punctuation if c != "/"]
    for token in remove_tokens:
        label = label.replace(token, "")
        pred = pred.replace(token, "")
    label = label.lower().replace("/", " ")
    pred = pred.lower().replace("/", " ")
    label_words = set(label.split(" "))
    pred_words = set(pred.split(" "))
    intersection = label_words.intersection(pred_words)
    union = label_words.union(pred_words)
    return len(intersection) / len(union) if union else 0.0


def evaluate_entry(
    labels: List[str], preds: List[str], substr_ok: bool
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for idx, pred in enumerate(preds):
        if not isinstance(pred, str):
            preds[idx] = str(pred)
    if len(labels) == 0:
        fp += len(preds)
    else:
        for ans in labels:
            if not ans:
                continue
            match_found = any(
                get_jaccard(ans, pred) >= IOU_THRESH or (substr_ok and ans in pred)
                for pred in preds
            )
            if match_found:
                tp += 1
            else:
                fn += 1
        for pred in preds:
            match_found = any(
                get_jaccard(ans, pred) >= IOU_THRESH or (substr_ok and ans in pred)
                for ans in labels
                if ans
            )
            if not match_found:
                fp += 1
    return tp, fp, fn


def handle_empty_preds(preds):
    if preds is None or (isinstance(preds, str) and preds in ("", " ", "null", "None")):
        return []
    elif isinstance(preds, float) and np.isnan(preds):
        return []
    if not isinstance(preds, (list, np.ndarray)):
        return [preds]
    return preds


def calculate_metrics(
    results: List[Dict[str, Any]], ground_truth: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Any]:
    tp = fp = fn = 0
    category_names = [cat["Category"] for cat in CUAD_CATEGORIES]
    gt_lower = {k.lower(): k for k in ground_truth}
    for result in results:
        contract_id = unicodedata.normalize("NFC", result.get("contract_id", ""))
        if contract_id not in ground_truth:
            contract_id = gt_lower.get(contract_id.lower(), contract_id)
        if contract_id not in ground_truth:
            print(f"Warning: Contract {contract_id} not in ground truth")
            continue
        labels = ground_truth[contract_id]
        for category in category_names:
            substr_ok = "Parties" in category
            label_spans = labels.get(category, [])
            pred_spans = handle_empty_preds(result.get(category, []))
            entry_tp, entry_fp, entry_fn = evaluate_entry(
                label_spans, pred_spans, substr_ok
            )
            tp += entry_tp
            fp += entry_fp
            fn += entry_fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

SUMMARY_FIELDNAMES = [
    "batch_size",
    "run_id",
    "fields_per_call",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "false_negatives",
] + SHARED_SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CUAD batch size × fields-per-call experiments"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, dest="output_dir_name")
    parser.add_argument("--num-contracts", type=int, default=100, dest="num_contracts")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fields-per-call",
        nargs="+",
        type=int,
        default=[41],
        dest="fields_per_call",
        help=f"Fields per LLM call to sweep (1–{_N_FIELDS}). Multiple values allowed, e.g. --fields-per-call 1 5 10 20 41",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 5, 10], dest="batch_sizes"
    )
    parser.add_argument("--direct-provider", action="store_true", default=False)
    parser.add_argument("--use-vertex", action="store_true", default=False)
    parser.add_argument("--include-thoughts", action="store_true", default=False)
    parser.add_argument("--disable-cache", action="store_true", default=False)
    parser.add_argument("--reasoning-effort", default=None)
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
    parser.add_argument("--plain-json", action="store_true", default=False, dest="plain_json",
                        help="Use json_object response format instead of strict json_schema (avoids Bedrock grammar-size limit)")
    parser.add_argument(
        "--run-id", type=int, default=1, dest="run_id",
        help="Run identifier (default: 1). run_id > 1 appends to existing per-call CSVs instead of overwriting.",
    )
    args = parser.parse_args()

    for fpc in args.fields_per_call:
        assert (
            1 <= fpc <= _N_FIELDS
        ), f"--fields-per-call values must be 1–{_N_FIELDS}, got {fpc}"

    output_dir = Path(__file__).parent / "results" / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Fields per call: {args.fields_per_call}")
    print(f"Contracts: {args.num_contracts} ({args.split} split, seed={args.seed})")

    print("Loading ground truth...")
    ground_truth = load_ground_truth(
        num_contracts=args.num_contracts, split=args.split, seed=args.seed
    )
    print(f"Loaded ground truth for {len(ground_truth)} contracts")

    _dataset = load_cuad_data(split=args.split)
    _filtered, sampled_titles = sample_contracts(
        _dataset, args.num_contracts, args.seed
    )
    sampled_contracts_meta = []
    for title in sampled_titles:
        rows = [r for r in _filtered if r["title"] == title]
        if rows:
            sampled_contracts_meta.append(
                {"contract_id": rows[0]["id"], "title": title}
            )

    settings = {
        "model": args.model,
        "direct_provider": args.direct_provider,
        "use_vertex": args.use_vertex,
        "fields_per_call": args.fields_per_call,
        "prompt_order": args.prompt_order,
        "doc_format": args.doc_format,
        "reasoning_effort": args.reasoning_effort,
        "include_thoughts": args.include_thoughts,
        "disable_cache": args.disable_cache,
        "streaming": args.streaming,
        "plain_json": args.plain_json,
        "num_contracts": args.num_contracts,
        "split": args.split,
        "seed": args.seed,
        "batch_sizes": args.batch_sizes,
        "sampled_contracts": sampled_contracts_meta,
    }

    csv_output = output_dir / "batch_size_summary.csv"
    json_output = output_dir / "batch_size_summary.json"
    json_first_entry = init_summary_files(
        csv_output, json_output, SUMMARY_FIELDNAMES, settings
    )

    model = Model(args.model)
    contracts = load_contracts(args.num_contracts, split=args.split, seed=args.seed)

    for fields_per_call in args.fields_per_call:
        for batch_size in args.batch_sizes:
            print(f"\n{'=' * 80}")
            print(f"fields_per_call={fields_per_call}  batch_size={batch_size}")
            print(f"{'=' * 80}")

            executor = CUADExecutor(
                model,
                fields_per_call=fields_per_call,
                batch_size=batch_size,
                direct_provider=args.direct_provider,
                use_vertex=args.use_vertex,
                include_thoughts=args.include_thoughts,
                disable_cache=args.disable_cache,
                reasoning_effort=args.reasoning_effort,
                plain_json=args.plain_json,
                prompt_order=args.prompt_order,
                doc_format=args.doc_format,
                streaming=args.streaming,
            )

            run_id = args.run_id
            executor.run_id = run_id
            per_call_csv = (
                output_dir
                / f"fpc{fields_per_call}_batch_size_{batch_size}_per_call.csv"
            )
            if run_id == 1 or not per_call_csv.exists():
                with open(per_call_csv, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(
                        f, fieldnames=executor.per_call_all_fieldnames
                    ).writeheader()
            executor.per_call_csv_path = per_call_csv
            json_stem = (
                f"fpc{fields_per_call}_batch_size_{batch_size}_per_call"
                if run_id == 1
                else f"fpc{fields_per_call}_batch_size_{batch_size}_run_{run_id}_per_call"
            )
            executor.per_call_json_path = output_dir / f"{json_stem}.json"

            contract_results, batch_stats = executor.execute(contracts, verbose=True)
            usage = executor.get_usage_summary()
            metrics = calculate_metrics(contract_results, ground_truth)

            csv_row = {
                "batch_size": batch_size,
                "run_id": run_id,
                "fields_per_call": fields_per_call,
                "accuracy": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                **executor._make_shared_summary_csv_fields(batch_stats, usage),
            }

            _shared = executor._make_shared_summary_json_fields(batch_stats, usage)
            json_entry = {
                "batch_size": batch_size,
                "run_id": run_id,
                "fields_per_call": fields_per_call,
                "model_name": batch_stats.model_name,
                "total_contracts_processed": batch_stats.total_docs_processed,
                "total_latency_secs": _shared["total_latency_secs"],
                "total_llm_calls": _shared["total_llm_calls"],
                "metrics": {"accuracy": metrics["f1"], **metrics},
                "input_token_details": _shared["input_token_details"],
                "output_token_details": _shared["output_token_details"],
                "cost": _shared["cost"],
            }

            print(
                f"\nF1: {metrics['f1']:.4f}  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}"
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
