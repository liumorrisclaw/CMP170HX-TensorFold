#!/usr/bin/env python3
"""Six-line-window source-overlap checker — the provenance gate for every CUDA source.

Community rule this tool enforces (ashhart/TensorFold review of PR #466, 2026-10-07):
only code that is its author's own expression gets landed. Any file that re-expresses
an idea from another project must show zero substantive six-line-window overlap with
that project's source, and the receipt goes into the file's provenance header.

Method (matches the check that flagged the original #466 draft at 11 windows):
  1. extract the CUDA payload (a CUDA_SRC raw-string block from a .py file,
     or the whole file for .cu/.cuh/.cpp);
  2. normalize: drop blank/comment/docstring lines, collapse whitespace;
  3. slide a 6-line window over the candidate; a hit is a window whose 6 normalized
     lines appear verbatim (as a block) in the reference;
  4. ignore trivial hits — windows whose every line is a bare brace/`break;`-style
     token (listed in TRIVIAL) — they carry no expression.

Usage:
  python3 tools/overlap_check.py FILE [FILE ...]          # vs all of reference/
  python3 tools/overlap_check.py --all                    # the standard candidate set
Exit code 0 = no substantive overlap; 1 = overlaps found (list them, then rewrite
the source or document why each hit is unavoidable); 2 = usage error.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO / "reference"

# Standard candidate set: every CUDA source this project authored or re-expressed.
# (Upstream-inherited kernels — experts.cu, qmm*.cu etc. — are TensorFold's own and
# are checked against upstream on the fork side, not here.)
DEFAULT_CANDIDATES = [
    "TensorFold066/src/tensorfold/cuda/shm_allgather.py",
    "TensorFold066/src/tensorfold/cuda/_shm_allgather.cu",
]

TRIVIAL = {"{", "}", "};", "break;", "continue;", "return;", "else {", "#pragma unroll", ")"}

# CUDA intrinsics / control scaffolding: lines with no authorial expression. A
# window made ONLY of these (plus braces) is a structural idiom — every bounded
# spin loop in every CUDA codebase closes the same way — and is reported as a
# NOTE rather than a failure. Anything carrying an identifier beyond this set
# is a substantive hit and fails the gate.
SCAFFOLD_RE = re.compile(
    r"^(__nanosleep\(\d+\);|__syncthreads\(\);|__threadfence\w*\(\);|break;|continue;)$"
)

WINDOW = 6


def extract_cuda(text: str, path: Path) -> str:
    m = re.search(r'CUDA_SRC\s*=\s*r?"""(.*?)"""', text, re.S)
    return m.group(1) if m else text


def norm_lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        s = re.sub(r"\s+", " ", ln.strip())
        if not s or s.startswith(("//", "#", '"""', "*")):
            continue
        out.append(s)
    return out


def windows(lines: list[str]) -> list[tuple[str, ...]]:
    return [tuple(lines[i:i + WINDOW]) for i in range(len(lines) - WINDOW + 1)]


def is_trivial(w: tuple[str, ...]) -> bool:
    return all(x in TRIVIAL or len(x) <= 2 or SCAFFOLD_RE.match(x) for x in w)


def check(candidate: Path, reference: Path) -> tuple[int, int, list[tuple[str, ...]]]:
    """(substantive_hits, trivial_hits, substantive windows)."""
    cand = norm_lines(extract_cuda(candidate.read_text(errors="replace"), candidate))
    ref = norm_lines(reference.read_text(errors="replace"))
    refset = set(windows(ref))
    substantive, trivial = [], 0
    for w in windows(cand):
        if w not in refset:
            continue
        if is_trivial(w):
            trivial += 1
        else:
            substantive.append(w)
    return len(substantive), trivial, substantive


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--all":
        cands = [REPO / p for p in DEFAULT_CANDIDATES]
    else:
        cands = [Path(a) for a in args]
    refs = sorted(p for p in REFERENCE_DIR.iterdir() if p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"))
    if not refs:
        print(f"no reference files in {REFERENCE_DIR}", file=sys.stderr)
        return 2
    bad = 0
    for c in cands:
        if not c.exists():
            print(f"MISSING  {c}")
            bad = 1
            continue
        for r in refs:
            n, t, hits = check(c, r)
            if n == 0 and t == 0:
                tag = "CLEAN"
            elif n == 0:
                tag = f"CLEAN ({t} trivial note)"
            else:
                tag = f"OVERLAP {n}"
            print(f"{tag:22s} {c.relative_to(REPO) if c.is_relative_to(REPO) else c}  vs  {r.name}")
            for w in hits[:5]:
                print(f"           e.g. {w[0][:96]}")
            if n:
                bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv))
