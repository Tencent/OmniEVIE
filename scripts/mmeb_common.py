"""Arguments, scoring and result output shared by the MMEB scripts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from omnievie import DEFAULT_MODEL_ID, QueryGate, ThreeStageIndex


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--scoring", default="index", choices=("index", "maxsim"),
                        help="index: package index with the query gate; maxsim: exact int8 MaxSim")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)


def int8_record(vectors: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(vectors.float(), dim=-1)
    return torch.clamp(torch.round(unit * 127.0), -127, 127) / 127.0


def pad_records(records: list[torch.Tensor], device) -> tuple[torch.Tensor, torch.Tensor]:
    size = max(r.shape[0] for r in records)
    stack = torch.zeros(len(records), size, records[0].shape[1])
    mask = torch.zeros(len(records), size, dtype=torch.bool)
    for i, r in enumerate(records):
        stack[i, :r.shape[0]] = r
        mask[i, :r.shape[0]] = True
    return stack.to(device), mask.to(device)


def maxsim(query: torch.Tensor, stack: torch.Tensor, mask: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    q = F.normalize(query.float().to(stack.device), dim=-1)
    scores = []
    for start in range(0, stack.shape[0], chunk):
        sim = torch.einsum("ld,nkd->lnk", q, stack[start:start + chunk])
        sim = sim.masked_fill(~mask[start:start + chunk][None], float("-inf")).amax(dim=2)
        scores.append(sim.sum(dim=0))
    return torch.cat(scores)


class Scorer:
    """Picks the best candidate for a query, over all candidates or a given list."""

    def __init__(self, vectors: dict[str, torch.Tensor], args: argparse.Namespace):
        self.names = list(vectors)
        self.maxsim = args.scoring == "maxsim"
        self.gate = None
        if self.maxsim:
            self.position = {name: i for i, name in enumerate(self.names)}
            self.stack, self.mask = pad_records([int8_record(v) for v in vectors.values()], args.device)
        else:
            self.index = ThreeStageIndex()
            for name, v in vectors.items():
                self.index.add(name, v, "doc")
            if not args.no_gate:
                self.gate = QueryGate.from_pretrained(args.gate or args.weights)

    def best(self, query, candidates: list[str] | None = None) -> str | None:
        names = self.names if candidates is None else candidates
        if self.maxsim:
            stack, mask = self.stack, self.mask
            if candidates is not None:
                ids = torch.tensor([self.position[name] for name in candidates], device=stack.device)
                stack, mask = stack[ids], mask[ids]
            return names[int(torch.argmax(maxsim(query.vectors, stack, mask)))]
        weights = None if self.gate is None else self.gate.weights(
            query.vectors, query.vision_mask, query.aug_mask, query.has_text)
        top = self.index.search(query.vectors, weights=weights, candidates=candidates, topk=1,
                                n1=len(names), n2=len(names))
        return top[0][0] if top else None


def write_result(args: argparse.Namespace, scorer: Scorer, hits: list[bool], started: float, **fields) -> None:
    result = {"task": args.task, **fields, "hit@1": round(sum(hits) / len(hits), 4),
              "scoring": args.scoring, "query_gate": scorer.gate is not None,
              "seconds": round(time.time() - started)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
