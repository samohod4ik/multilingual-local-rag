"""Benchmark commands plus local index and search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataclasses import asdict

from multilingual_local_rag.adapters.filesystem import FilesystemAdapter
from multilingual_local_rag.audit import audit_tree
from multilingual_local_rag.evaluation.dataset import load_dataset
from multilingual_local_rag.evaluation.envelope import load_predictions
from multilingual_local_rag.evaluation.manifest import dataset_hash
from multilingual_local_rag.evaluation.metrics import evaluate_run
from multilingual_local_rag.index.builder import build_snapshot, load_chunks, read_current
from multilingual_local_rag.index.vector_cache import VectorCache
from multilingual_local_rag.providers.deterministic import HashEmbedder, OverlapReranker
from multilingual_local_rag.providers.pinned import PinnedEmbedder, PinnedReranker
from multilingual_local_rag.retrieval.hybrid import search_snapshot
from multilingual_local_rag.service.api import LoopbackServer, load_state


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


def _providers(profile: str, embedder_name: str) -> tuple[object | None, object | None, str]:
    if profile == "lexical":
        return None, None, "lexical"
    if embedder_name == "hash":
        return HashEmbedder(), OverlapReranker(), HashEmbedder.model_id
    return PinnedEmbedder(), PinnedReranker(), PinnedEmbedder.model_id


def _cmd_index(args: argparse.Namespace) -> int:
    embedder, _reranker, model_id = _providers(args.profile, args.embedder)
    snapshot_id = build_snapshot(
        FilesystemAdapter(args.source),
        args.data_root,
        embedder=embedder,  # type: ignore[arg-type]
        profile=args.profile,
        model_id=model_id,
    )
    print(snapshot_id)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    current = read_current(args.data_root)
    if current is None:
        print("no snapshot", file=sys.stderr)
        return 1
    chunks = load_chunks(args.data_root, current["snapshot_id"])
    embedder, reranker, model_id = _providers(args.profile, args.embedder)
    cache = None
    if embedder is not None:
        cache = VectorCache(Path(args.data_root) / "caches" / "vectors.sqlite")
    try:
        result = search_snapshot(
            chunks,
            args.query,
            profile=args.profile,
            embedder=embedder,  # type: ignore[arg-type]
            reranker=reranker,  # type: ignore[arg-type]
            cache=cache,
            model_id=model_id,
        )
    finally:
        if cache is not None:
            cache.close()
    body = asdict(result)
    print(json.dumps(body, ensure_ascii=False, sort_keys=True))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    current = read_current(args.data_root)
    print(json.dumps(current, sort_keys=True))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    state = load_state(args.data_root, profile=args.profile, embedder_name=args.embedder)
    server = LoopbackServer(args.host, args.port, state)
    server.serve_forever()
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
    index = sub.add_parser("index")
    index.add_argument("source", type=Path)
    index.add_argument("--data-root", type=Path, required=True)
    index.add_argument("--profile", choices=("lexical", "quality"), default="lexical")
    index.add_argument("--embedder", choices=("pinned", "hash"), default="pinned")
    index.set_defaults(func=_cmd_index)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--data-root", type=Path, required=True)
    search.add_argument("--profile", choices=("lexical", "quality"), default="lexical")
    search.add_argument("--embedder", choices=("pinned", "hash"), default="pinned")
    search.set_defaults(func=_cmd_search)
    status = sub.add_parser("status")
    status.add_argument("--data-root", type=Path, required=True)
    status.set_defaults(func=_cmd_status)
    serve = sub.add_parser("serve")
    serve.add_argument("--data-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--profile", choices=("lexical", "quality"), default="lexical")
    serve.add_argument("--embedder", choices=("pinned", "hash"), default="hash")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
