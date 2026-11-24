#!/usr/bin/env python3
"""Hit@1 on one MMEB video task (32 of the 64 saved frames, at most 14,400 tokens).

t2v: manifest columns qry_text, qry_inst, pos_id; videos in <frame-root>/<pos_id>/.
v2t: manifest columns qry_id, qry_inst, pos_label; videos in <frame-root>/<qry_id>/.
     A cand_labels column ranks each query over its own list (SmthSmthV2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from mmeb_common import Scorer, add_common_args, write_result
from omnievie import OmniEVIE

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_frames(directory: Path) -> list[Image.Image]:
    names = sorted((p.name for p in directory.iterdir() if p.suffix.lower() in FRAME_SUFFIXES), key=_natural_key)
    frames = []
    for name in names:
        with Image.open(directory / name) as image:
            frames.append(image.convert("RGB"))
    return frames


def in_chunks(pool: ThreadPoolExecutor, fn, items: list, workers: int):
    for start in range(0, len(items), workers):
        chunk = items[start:start + workers]
        yield from zip(chunk, pool.map(fn, chunk))


def cache_file(cache: Path, name: str) -> Path:
    return cache / (hashlib.sha1(name.encode("utf-8")).hexdigest() + ".pt")


def candidate_names(rows: list[dict], key: str, labels: Path | None) -> list[str]:
    if labels is not None:
        names = json.loads(labels.read_text(encoding="utf-8"))
        unknown = {row[key] for row in rows} - set(names)
        if unknown:
            raise SystemExit(f"{len(unknown)} positive labels are not in {labels}")
        return names
    if "cand_labels" in rows[0]:
        return sorted({label for row in rows for label in row["cand_labels"]})
    return sorted({row[key] for row in rows})


def encode_videos(model: OmniEVIE, names: list[str], args, pool) -> dict[str, torch.Tensor]:
    vectors = {}
    todo = names if args.cache is None else [n for n in names if not cache_file(args.cache, n).exists()]
    clips = in_chunks(pool, lambda n: load_frames(args.frame_root / n), todo, args.workers)
    for count, (name, frames) in enumerate(clips, 1):
        vectors[name] = model.encode_clip(frames, protocol="mmeb")
        if args.cache is not None:
            target = cache_file(args.cache, name)
            partial = target.with_suffix(".partial")
            torch.save(vectors[name], partial)
            partial.replace(target)  # a stopped run never leaves a truncated cache file
        if count % 500 == 0:
            print(f"[{args.task}] {count}/{len(todo)} candidates", flush=True)
    if args.cache is not None:
        vectors.update({
            n: torch.load(cache_file(args.cache, n), map_location="cpu", weights_only=True)
            for n in names if n not in vectors
        })
    return {n: vectors[n] for n in names}


def encode_queries(model: OmniEVIE, rows: list[dict], args, pool) -> list:
    if args.direction == "t2v":
        return [model.encode_query(text=row["qry_text"], target="video",
                                   instruction=(row["qry_inst"] or "").strip(), max_chars=None) for row in rows]
    clips = in_chunks(pool, lambda r: load_frames(args.frame_root / r["qry_id"]), rows, args.workers)
    return [model.encode_query(video=frames, protocol="mmeb", instruction=(row["qry_inst"] or "").strip())
            for row, frames in clips]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--direction", required=True, choices=("t2v", "v2t"))
    parser.add_argument("--labels", type=Path, default=None,
                        help="v2t: JSON list of all labels (default: distinct pos_label)")
    parser.add_argument("--cache", type=Path, default=None, help="t2v: per-candidate vector cache")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="t2v: cache candidates K, K+N, ... and exit; rerun without --shard to score")
    add_common_args(parser)
    args = parser.parse_args()
    if args.shard and (args.cache is None or args.direction != "t2v"):
        raise SystemExit("--shard needs --cache and --direction t2v")
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=True)

    started = time.time()
    key = "pos_id" if args.direction == "t2v" else "pos_label"
    rows = pq.read_table(args.manifest).to_pylist()
    local = "cand_labels" in rows[0]
    names = candidate_names(rows, key, args.labels)
    print(f"[{args.task}] {len(rows)} queries, {len(names)} candidates", flush=True)

    model = OmniEVIE.from_pretrained(args.weights, device=args.device)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.shard:
            k, n = (int(part) for part in args.shard.split("/"))
            encode_videos(model, names[k::n], args, pool)
            print(f"[{args.task}] shard {args.shard} cached; rerun without --shard to score", flush=True)
            return
        if args.direction == "t2v":
            vectors = encode_videos(model, names, args, pool)
        else:
            vectors = {label: model.encode_text(label) for label in names}
        print(f"[{args.task}] candidates encoded in {time.time() - started:.0f}s", flush=True)
        scorer = Scorer(vectors, args)
        queries = encode_queries(model, rows, args, pool)

    hits = [scorer.best(query, row["cand_labels"] if local else None) == row[key] for query, row in zip(queries, rows)]
    write_result(args, scorer, hits, started, direction=args.direction, queries=len(rows), candidates=len(names))


if __name__ == "__main__":
    main()
