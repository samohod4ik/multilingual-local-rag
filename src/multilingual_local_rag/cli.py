"""Benchmark validation and evaluation commands. No retrieval implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multilingual_local_rag.audit import audit_tree
from multilingual_local_rag.evaluation.dataset import load_dataset
from multilingual_local_rag.evaluation.envelope import load_predictions
from multilingual_local_rag.evaluation.manifest import dataset_hash
from multilingual_local_rag.evaluation.metrics import evaluate_run


def _cmd_validate(args: argparse.Namespace) -> int:
    load_dataset(args.benchmark)
    print(dataset_hash(args.benchmark))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    digest = dataset_hash(args.benchmark)
    _run_id, predictions = load_predictions(
        args.predictions, dataset_hash=digest, config_hash=args.config_hash
    )
    dataset = load_dataset(args.benchmark)
    summary = evaluate_run(dataset, predictions)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    problems = audit_tree(args.root)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("audit-ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlrag")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-benchmark")
    validate.add_argument("benchmark", type=Path)
    validate.set_defaults(func=_cmd_validate)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("benchmark", type=Path)
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--config-hash", required=True)
    evaluate.set_defaults(func=_cmd_evaluate)
    audit = sub.add_parser("audit-fixtures")
    audit.add_argument("root", type=Path)
    audit.set_defaults(func=_cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
