#!/usr/bin/env python3
"""Prebuild the heavy CUDA extensions for the v0.6.6 combined tree, CPU-only (nvcc needs no GPU).

Run on <GPU_HOST>: nohup python3 prebuild066.py > logs/prebuild066.log 2>&1 &
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/<user>/tensorfold-poc/TensorFold066/src")
os.environ["TENSORFOLD_MIN_CAPABILITY"] = "8.0"
os.environ["TORCH_EXTENSIONS_DIR"] = "/home/<user>/tensorfold-poc/tf_ext066"
os.environ.setdefault("PYTHONPATH",
                      "/home/<user>/glm53-recipe/venv/lib/python3.12/site-packages:/home/<user>/tensorfold-poc/TensorFold066/src")

from tensorfold.cuda.build import load

HERE = Path("/home/<user>/tensorfold-poc/TensorFold066/src/tensorfold")
GLM = HERE / "families" / "glm5_next" / "cuda"

JOBS = [
    ("tensorfold_qmm_v5",
     [HERE / "cuda" / "kernels" / f for f in
      ("qmm.cpp", "qmm.cu", "qmm_group.cu", "qmm_prefill.cu", "qmm_prefill8.cu")],
     ["-O3", "-lineinfo"]),
    ("tensorfold_exl3_experts_v1",
     [HERE / "cuda" / "exl3" / f for f in
      ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")],
     ["-O3", "-lineinfo"]),
    ("tensorfold_glm_kda_v2",
     [GLM / "kda.cpp", GLM / "kda.cu"],
     ["-O3", "--fmad=false"]),
    ("tensorfold_glm_exl3_v1",
     [GLM / "exl3.cpp", GLM / "exl3.cu"],
     ["-O3", "-lineinfo"]),
]

for name, srcs, flags in JOBS:
    print("building", name, flush=True)
    load(name=name, sources=[str(s) for s in srcs], extra_cuda_cflags=flags, verbose=False)
    print("done", name, flush=True)
print("PREBUILD_COMPLETE", flush=True)
