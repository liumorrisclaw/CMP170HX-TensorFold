#!/usr/bin/env python3
"""TensorFold 4-rank GLM A/B bench — MiaAI-Lab protocol, identical to the 2026-10-05 tf4 receipts.

Streaming, T=0, thinking off, 400-token output, warmup 32, 3 runs, median.
tok_s = (completion_tokens - 1) / decode_s   (decode_s = last - first SSE chunk).
Captures /health before+after for drafted/accepted counters; optional exactness
probe (draft:true vs draft:false SHA256, the exact-decoding invariant).

Usage: tf4_ab_bench.py --profile dflash [--exactness] [--max-tokens 400]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8899"
MODEL = "glm53-tr3-4bpw"  # overridable via --model (supports @policy suffixes)
PROMPTS = {
    "structured": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.",
    "coding": ("Write a Python function named clamp_range that takes a list of ints and "
               "returns a new list with each value clamped to [0, 50]. Include a short docstring."),
    "prose": ("Write a detailed step-by-step explanation of how a hash map works, "
              "including collision handling, resizing, and time complexity. Be thorough."),
}


def health() -> dict:
    with urllib.request.urlopen(BASE + "/health", timeout=15) as r:
        return json.loads(r.read().decode())


def ask_stream(prompt: str, max_tokens: int, draft: bool, model: str = MODEL) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "draft": draft,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content_parts: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    content_parts.append(piece)
    end = time.time()
    content = "".join(content_parts)
    completion_tokens = usage.get("completion_tokens") or 0
    wall = end - t0
    decode_s = (end - t0 - ttft) if ttft is not None else wall
    return {
        "ttft_s": ttft,
        "wall_s": wall,
        "decode_s": decode_s,
        "tok_s": (completion_tokens - 1) / decode_s if completion_tokens and decode_s > 0 else 0.0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "finish": "ok" if completion_tokens else "empty",
        "content": content,
        "sha": hashlib.sha256(content.encode()).hexdigest()[:16],
    }


def bench_prompt(name: str, prompt: str, runs: int, max_tokens: int, model: str = MODEL) -> dict:
    warm = ask_stream(prompt, 32, draft=True, model=model)
    recs = []
    for i in range(runs):
        r = ask_stream(prompt, max_tokens, draft=True, model=model)
        r.pop("content")
        recs.append(r)
        print(f"  [{name}] run {i+1}/{runs}: {r['tok_s']:.1f} tok/s "
              f"({r['completion_tokens']} tok, ttft {r['ttft_s']:.2f}s)", flush=True)
    tok = sorted(r["tok_s"] for r in recs)
    return {
        "warmup_tok_s": warm["tok_s"],
        "runs": recs,
        "tok_s_median": statistics.median(tok),
        "ttft_median_s": statistics.median(r["ttft_s"] for r in recs),
        "completion_tokens_median": statistics.median(r["completion_tokens"] for r in recs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--exactness", action="store_true",
                    help="after the bench: draft:true vs draft:false greedy SHA256 check")
    ap.add_argument("--only", choices=list(PROMPTS), help="bench a single prompt")
    ap.add_argument("--model", default=MODEL, help="model id, @suffix selects a draft policy")
    ap.add_argument("--tag", default="", help="output filename tag")
    args = ap.parse_args()

    h0 = health()
    rec = {
        "phase": f"tf4ab_{args.profile}",
        "ts": time.time(),
        "max_tokens": args.max_tokens,
        "health_before": h0,
        "model": args.model,
        "prompts": {},
    }
    names = [args.only] if args.only else list(PROMPTS)
    for name in names:
        print(f"[bench] {name} (profile={args.profile})", flush=True)
        rec["prompts"][name] = bench_prompt(name, PROMPTS[name], args.runs, args.max_tokens, model=args.model)

    if args.exactness:
        print("[exactness] draft:true vs draft:false (structured, greedy)", flush=True)
        d = ask_stream(PROMPTS["structured"], 400, draft=True, model=args.model)
        s = ask_stream(PROMPTS["structured"], 400, draft=False, model=args.model)
        rec["exactness"] = {
            "draft_sha": d["sha"], "serial_sha": s["sha"],
            "draft_tokens": d["completion_tokens"], "serial_tokens": s["completion_tokens"],
            "match": d["sha"] == s["sha"] and d["content"] == s["content"],
        }
        print(f"[exactness] match={rec['exactness']['match']} "
              f"({d['completion_tokens']} vs {s['completion_tokens']} tok)", flush=True)

    h1 = health()
    rec["health_after"] = h1
    for k in ("drafted_total", "accepted_total", "rounds_total", "requests_total"):
        rec.setdefault("counters", {})[k] = (h1.get(k), h0.get(k))
    drafted = (h1.get("drafted_total") or 0) - (h0.get("drafted_total") or 0)
    accepted = (h1.get("accepted_total") or 0) - (h0.get("accepted_total") or 0)
    rounds = (h1.get("rounds_total") or 0) - (h0.get("rounds_total") or 0)
    rec["acceptance"] = {
        "drafted_delta": drafted, "accepted_delta": accepted, "rounds_delta": rounds,
        "accept_ratio": accepted / drafted if drafted else None,
        "tok_per_step": (rounds + accepted) / rounds if rounds else None,
    }

    out = Path.home() / 'bench' / f'tf4ab_{args.profile}_{args.tag or (args.only or 'full')}.json'
    out.write_text(json.dumps(rec, indent=2))
    print(f"[bench] wrote {out}", flush=True)
    summary = {n: round(rec["prompts"][n]["tok_s_median"], 1) for n in rec["prompts"]}
    print(f"RESULT profile={args.profile} tok_s_median={summary} "
          f"accept_ratio={rec['acceptance']['accept_ratio']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
