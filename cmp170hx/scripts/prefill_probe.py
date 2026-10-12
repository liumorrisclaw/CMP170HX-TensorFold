#!/usr/bin/env python3
"""Standard cold-prefill probe — every point prints its CONTEXT LENGTH.

Unique prompt per shot (defeats the prefix cache; a repeated prompt measures
cache hits, not prefill -- the 2026-10-11 lesson). Usage:

    TF_BENCH_BASE=http://127.0.0.1:8000 python3 prefill_probe.py [ lengths... ]
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("TF_BENCH_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("TF_BENCH_MODEL", "glm53-tr3-4bpw")
UNIT = ("Audit line {i}: the writer pins pages, the reader joins stripes, "
        "the checker signs every row. ")


def ask(prompt: str) -> float:
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        json.loads(resp.read())
    return time.time() - t0


def main() -> int:
    lengths = [int(x) for x in sys.argv[1:]] or [1500, 6000, 16000, 31000]
    print(f"{'context (tok)':>14}  {'prefill (tok/s)':>16}  {'wall (ms)':>10}", flush=True)
    for target in lengths:
        reps = max(1, target // 45)
        body = "".join(UNIT.format(i=i) for i in range(reps))
        n = int(reps * 44.6)
        ask(f"warm-{target}-{time.time_ns()}: " + body)   # different nonce: warms nothing shared
        wall = ask(f"cold-{target}-{time.time_ns()}: " + body)   # measured, unique
        print(f"{n:>14}  {n / wall:>16.0f}  {wall * 1000:>10.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
