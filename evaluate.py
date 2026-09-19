import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

import baseline_rag
from rag import DEFAULT_EMBEDDER, IntegratedQA, ROOT


REQUIRED = {"case_id", "question", "case_type", "expected_doc_ids",
            "doc_match", "expected_answer"}
KINDS = {"answerable", "near_duplicate", "multi_doc", "conflict", "unanswerable"}


def load_cases(path, known_doc_ids):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError("Evaluation CSV is missing required columns")
        cases = list(reader)
    if not cases:
        raise ValueError("Evaluation CSV is empty")
    seen = set()
    for case in cases:
        identifier = case["case_id"]
        expected = set(filter(None, case["expected_doc_ids"].split("|")))
        if not identifier or identifier in seen or not case["question"] or case["case_type"] not in KINDS:
            raise ValueError(f"Invalid or duplicate case: {identifier}")
        seen.add(identifier)
        if expected - known_doc_ids:
            raise ValueError(f"Unknown expected document in {identifier}")
        if case["case_type"] == "unanswerable":
            if expected or case["doc_match"] != "none":
                raise ValueError(f"Invalid unanswerable reference: {identifier}")
        elif not expected or case["doc_match"] not in {"all", "any"}:
            raise ValueError(f"Invalid answerable reference: {identifier}")
        case["expected_set"] = expected
    return cases


def passed_case(case, returned_doc_ids):
    policy = case["doc_match"]
    if policy == "none":
        return not returned_doc_ids
    if policy == "any":
        return bool(case["expected_set"] & returned_doc_ids)
    return case["expected_set"] <= returned_doc_ids


def summarize(cases, results):
    groups = defaultdict(list)
    for case, result in zip(cases, results):
        kind = case["case_type"]
        expected_status = "abstain" if kind == "unanswerable" else (
            "conflict" if kind == "conflict" else "answered")
        passed = result["status"] == expected_status and passed_case(case, set(result["sources"]))
        groups[kind].append(passed)
        print(f"{case['case_id']:<5} {kind:<15} {result['status']:<10} "
              f"{','.join(result['sources']):<17} {'PASS' if passed else 'FAIL'}")
        if result.get("show_answer"):
            print("      " + result["answer"])
    print("\nSummary (status and source only)")
    for label, kinds in (("Single-document", ("answerable", "near_duplicate")),
                         ("Multi-document", ("multi_doc",)),
                         ("Conflict", ("conflict",)),
                         ("Abstention", ("unanswerable",))):
        values = [value for kind in kinds for value in groups[kind]]
        if values:
            print(f"{label}: {sum(values)}/{len(values)}")
    print("Answer content requires manual review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=ROOT / "evaluation.csv")
    parser.add_argument("--system", choices=("final", "baseline"), default="final")
    parser.add_argument("--show-answers", action="store_true")
    args = parser.parse_args()

    docs = baseline_rag.load_docs(ROOT / "corpus.jsonl")
    cases = load_cases(args.eval, {doc["id"] for doc in docs})
    if args.system == "final":
        engine = IntegratedQA()
        results = [engine.answer(case["question"]) for case in cases]
    else:
        model = SentenceTransformer(str(DEFAULT_EMBEDDER), local_files_only=True)
        chunks, vectors = baseline_rag.build_index(docs, model)
        results = []
        for case in cases:
            hit, _ = baseline_rag.retrieve(case["question"], chunks, vectors, model)
            results.append({"status": "answered", "sources": [hit["doc_id"]],
                            "answer": f"[{hit['doc_id']}] {hit['text']}"})
    if args.show_answers:
        for result in results:
            result["show_answer"] = True
    summarize(cases, results)


if __name__ == "__main__":
    main()
