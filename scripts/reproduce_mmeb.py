#!/usr/bin/env python3
"""Hit@1 on one MMEB image-query task.

Parquet columns: qry_text, qry_img_path, qry_inst, tgt_img_path (candidate 0 is the positive).
Each query is ranked over its own candidate list.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from mmeb_common import Scorer, add_common_args, write_result
from omnievie import OmniEVIE, pooling


def open_rgb(root: Path, relative: str) -> Image.Image:
    with Image.open(root / relative) as image:
        return image.convert("RGB")


def encode_candidates(model: OmniEVIE, names: list[str], root: Path, workers: int) -> dict:
    vectors = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(names), workers):
            chunk = names[start:start + workers]
            for name, image in zip(chunk, pool.map(lambda n: open_rgb(root, n), chunk)):
                [(tokens, grid)] = model.image_tokens([image])
                vectors[name] = pooling.image(tokens, grid)
    return vectors


def encode_query(model: OmniEVIE, row: dict, root: Path, instructions: str):
    image = open_rgb(root, row["qry_img_path"]) if row["qry_img_path"] else None
    text = row["qry_text"] or None
    if instructions == "scene":
        return model.encode_query(text=text, image=image, target="image")
    return model.encode_query(text=text, image=image, target="image",
                              instruction=(row["qry_inst"] or "").strip(), max_chars=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--instructions", default="official", choices=("official", "scene"),
                        help="official: the parquet's qry_inst; scene: packaged instructions")
    add_common_args(parser)
    args = parser.parse_args()

    started = time.time()
    rows = pq.read_table(args.parquet, columns=["qry_text", "qry_img_path", "qry_inst", "tgt_img_path"]).to_pylist()
    if any(not row["tgt_img_path"] for row in rows):
        raise SystemExit("every row needs a non-empty candidate list")
    names = sorted({name for row in rows for name in row["tgt_img_path"]})
    print(f"[{args.task}] {len(rows)} queries, {len(names)} candidate images", flush=True)

    model = OmniEVIE.from_pretrained(args.weights, device=args.device)
    scorer = Scorer(encode_candidates(model, names, args.image_root, args.workers), args)
    hits = [scorer.best(encode_query(model, row, args.image_root, args.instructions), row["tgt_img_path"])
            == row["tgt_img_path"][0] for row in rows]
    write_result(args, scorer, hits, started, instructions=args.instructions, queries=len(rows),
                 candidates=len(names))


if __name__ == "__main__":
    main()
