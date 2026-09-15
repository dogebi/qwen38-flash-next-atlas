#!/usr/bin/env python3
"""Parallel header measurement: same output as measure.py, 8 shards in flight.

usage: HF_MEASURE_DIR=/tmp/qn-measure python3 measure_fast.py <repo> [<repo> ...]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", HERE / "measure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

OUT = m.OUT
WORKERS = int(os.environ.get("HF_MEASURE_WORKERS", "8"))


def measure(repo: str) -> dict:
    files = sorted((e for e in m.tree(repo) if e["type"] == "file" and e["path"].endswith(".safetensors")),
                   key=lambda e: e["path"])
    print(f"== {repo}: {len(files)} shards · {WORKERS} workers", flush=True)
    results: dict[str, dict] = {}
    done = 0

    def one(e):
        return e["path"], m.header(repo, e["path"])

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for path, h in ex.map(one, files):
            done += 1
            if done % 25 == 0 or done == len(files):
                print(f"   ... {done}/{len(files)} shards", flush=True)
            results[path] = h

    tensors: dict[str, dict] = {}
    for path in sorted(results):
        for name, t in results[path].items():
            if name == "__metadata__":
                continue
            off = t["data_offsets"]
            if name in tensors:
                raise SystemExit(f"duplicate tensor {name} in {repo}")
            tensors[name] = {"dtype": t["dtype"], "shape": t["shape"],
                             "bytes": off[1] - off[0], "shard": path}
    payload = sum(t["bytes"] for t in tensors.values())
    disk = sum((e.get("lfs") or {}).get("size") or e.get("size") or 0 for e in files)
    out = {"repo": repo, "shards": len(files), "tensor_count": len(tensors),
           "payload_bytes": payload, "disk_bytes": disk, "tensors": tensors}
    (OUT / (repo.replace("/", "__") + ".measured.json")).write_text(json.dumps(out, indent=1))
    print(f"   {repo}: payload {payload / 1e9:.4f} GB · disk {disk / 1e9:.4f} GB · tensors {len(tensors)}", flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        measure(repo)
