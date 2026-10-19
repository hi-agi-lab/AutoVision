#!/usr/bin/env python
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-node NCCL smoke test for the training launcher.

Run on every node (same command, coordinated by torchrun):

    torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=<node0>:29500 \
        scripts/nccl_smoke.py --timeout 120

It verifies that the process group initializes, all ranks agree on the
world size, and an all-reduce round-trips. Exit code 1 on any mismatch or
timeout; this is the first check before a multi-node training job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=120, help="seconds")
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args(argv)

    if not torch.distributed.is_available():
        print("torch.distributed is unavailable.", file=sys.stderr)
        return 2
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=args.backend,
            timeout=dt.timedelta(seconds=args.timeout),
        )
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if not torch.cuda.is_available():
            print("NCCL backend needs CUDA.", file=sys.stderr)
            return 2
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    tensor = torch.ones(1, device=device) * (rank + 1)
    torch.distributed.all_reduce(tensor)
    expected = world * (world + 1) / 2
    actual = float(tensor.item())
    ok = abs(actual - expected) < 1e-3
    if rank == 0:
        print(
            f"backend={backend} world_size={world} device={device} "
            f"all_reduce={actual} expected={expected} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
    torch.distributed.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
