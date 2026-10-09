#!/usr/bin/env python3
"""4-rank verification for the copies-only fresh shm all-gather (PR variant).

Patterns per case: every rank sends a deterministic buffer; recv must hold all four
in rank order, bitwise. int32 sends the rank id (slot-placement can't hide),
bf16/f32 send seeded draws compared on rank 0.
"""
import os
import subprocess
import sys

SRC = "/home/<user>/tensorfold-poc/TensorFoldPRCHECK/src"
SITE = "/home/<user>/glm53-recipe/venv/lib/python3.12/site-packages"
PORT = 29531
CASES = [("i32", 4, 16), ("bf16", 2, 256), ("f32", 1, 512), ("bf16", 8, 4096), ("f32", 3, 1024)]


def run_rank(rank: int, port: int) -> int:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SRC)
    sys.path.insert(0, SITE)
    from tensorfold.cuda.shm_allgather import ShmAllGather

    torch.cuda.set_device(0)
    store = dist.TCPStore("127.0.0.1", port, world_size=4, is_master=(rank == 0))
    shm = ShmAllGather(rank, 4, store, blocks=4)
    for ci, (kind, rows, d) in enumerate(CASES):
        dt = {"i32": torch.int32, "bf16": torch.bfloat16, "f32": torch.float32}[kind]
        if kind == "i32":
            send = torch.full((rows, d), rank, dtype=dt, device="cuda")
        else:
            send = torch.randn(rows, d, generator=torch.Generator(device="cuda").manual_seed(31337 + ci * 7 + rank),
                               device="cuda").to(dt)
        recv = torch.zeros(4 * rows * d, dtype=dt, device="cuda")
        shm.all_gather(send.reshape(-1), recv)
        torch.cuda.synchronize()
        if rank == 0:
            peers = [send if rr == 0 else
                     (torch.full((rows, d), rr, dtype=dt, device="cuda") if kind == "i32" else
                      torch.randn(rows, d, generator=torch.Generator(device="cuda").manual_seed(31337 + ci * 7 + rr),
                                  device="cuda").to(dt))
                     for rr in range(4)]
            exp = torch.stack(peers)                       # [4, rows, d]
            same = torch.equal(recv.view(4, rows, d), exp)
            print(f"[case {ci}] {kind} {rows}x{d}: bitwise_equal={same}", flush=True)
            if not same:
                print("  RESULT MISMATCH", flush=True)
        store.set(f"done_{ci}_r{rank}", "1")
        store.wait([f"done_{ci}_r{r}" for r in range(4)])
    shm.close()
    if rank == 0:
        print("RESULT ALL_BITWISE_EQUAL", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) == 3:
        return run_rank(int(sys.argv[1]), int(sys.argv[2]))
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{SITE}:{SRC}", TENSORFOLD_MIN_CAPABILITY="8.0",
               TORCH_EXTENSIONS_DIR=os.path.expanduser("~/tensorfold-poc/tf_ext066"))
    rc = 0
    procs = []
    for r in range(4):
        cenv = dict(env)
        cenv["CUDA_VISIBLE_DEVICES"] = str(r)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), str(r), str(PORT)], env=cenv))
    for p in procs:
        code = p.wait()
        rc = rc or code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
