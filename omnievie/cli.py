"""Command line interface.

    python -m omnievie.cli index  OUT.pt  FILE_OR_DIR [FILE_OR_DIR ...]
    python -m omnievie.cli search INDEX.pt --text "question" --target text
    python -m omnievie.cli search INDEX.pt --image query.jpg --target image
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .encoder import (
    DEFAULT_MODEL_ID,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
    OmniEVIE,
)
from .gate import QueryGate
from .index import N1_DEFAULT, N2_DEFAULT, ThreeStageIndex

SUFFIXES = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES))
        else:
            files.append(path)
    return files


def cmd_index(args: argparse.Namespace) -> None:
    model = OmniEVIE.from_pretrained(args.weights, device=args.device)
    index = ThreeStageIndex()
    for path in _collect(args.inputs):
        modality, vectors = model.encode_document(path)
        index.add(str(path), vectors, modality)
        print(f"{modality:6s} {vectors.shape[0]:4d} vectors  {path}", flush=True)
    index.save(args.out)
    print(f"wrote {len(index)} documents to {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    if not (args.text or args.image):
        raise SystemExit("give --text or --image")
    index = ThreeStageIndex.load(args.index, n1=args.n1, n2=args.n2)
    model = OmniEVIE.from_pretrained(args.weights, device=args.device)
    query = model.encode_query(text=args.text, image=args.image, target=args.target)
    weights = None
    if not args.no_gate:
        weights = QueryGate.from_pretrained(args.weights).weights(
            query.vectors, query.vision_mask, query.aug_mask, query.has_text
        )
    for doc_id, score in index.search(query.vectors, weights=weights, modality=args.target, topk=args.top):
        print(f"{score:9.3f}  {doc_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnievie")
    parser.add_argument(
        "--weights",
        default=DEFAULT_MODEL_ID,
        help=f"local weights directory or Hugging Face repo ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument("--device", default="cuda")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="encode documents and write an index file")
    p_index.add_argument("out")
    p_index.add_argument("inputs", nargs="+")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="search an index file")
    p_search.add_argument("index")
    p_search.add_argument("--text")
    p_search.add_argument("--image")
    p_search.add_argument("--target", default="text", choices=("text", "image", "video"))
    p_search.add_argument("--top", type=int, default=10)
    p_search.add_argument("--n1", type=int, default=N1_DEFAULT)
    p_search.add_argument("--n2", type=int, default=N2_DEFAULT)
    p_search.add_argument("--no-gate", action="store_true", help="uniform query weights")
    p_search.set_defaults(func=cmd_search)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
