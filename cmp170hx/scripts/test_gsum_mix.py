#!/usr/bin/env python3
"""4-rank bit-exactness A/B for allgather_sum_mix (handoff Task E full version).

Case = one (rows, D) shape on the deployed shm config (world=4, blocks=4):
  A: shm.all_gather_sum(part) -> spart;  glue.hc_mix(x, outA, spart, post, comb)   [deployed path]
  B: shm.all_gather_sum_mix(part, xB, post, comb)                                  [folded kernel]
Rank 0 requires torch.equal(outA, xB) bitwise for every case, else exits 1.

Run on <GPU_HOST>:  python3 test_gsum_mix.py            (spawns 4 GPU children, ranks 0-3)
"""
import os
import subprocess
import sys

SRC = "/home/<user>/tensorfold-poc/TensorFold066/src"
SITE = "/home/<user>/glm53-recipe/venv/lib/python3.12/site-packages"
PORT = 29517
CASES = [(4, 4096), (1, 4096), (8, 4096), (3, 4096), (2, 4096), (4, 256), (6, 1024)]


def run_rank(rank: int, port: int) -> int:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SRC)
    sys.path.insert(0, SITE)
    from tensorfold.cuda.shm_allgather import ShmAllGather
    from tensorfold.families.glm5_next.cuda import glue

    torch.cuda.set_device(0)
    store = dist.TCPStore("127.0.0.1", port, world_size=4, is_master=(rank == 0))
    shm = ShmAllGather(rank, 4, store, blocks=4)
    bad = 0
    for ci, (rows, d) in enumerate(CASES):
        g = torch.Generator(device="cuda").manual_seed(20261009 + ci)          # shared tensors
        gp = torch.Generator(device="cuda").manual_seed(777 + rank * 100 + ci)  # per-rank partial
        part = torch.randn(rows, d, generator=gp, device="cuda").to(torch.bfloat16)
        x = torch.randn(rows, 4 * d, generator=g, device="cuda").to(torch.bfloat16)
        post = torch.rand(rows, 4, generator=g, device="cuda")
        comb = torch.rand(rows, 16, generator=g, device="cuda")

        spart = torch.empty(rows, d, dtype=torch.bfloat16, device="cuda")
        shm.all_gather_sum(part.reshape(-1), spart.reshape(-1))
        out_a = torch.empty_like(x)
        glue.hc_mix(x, out_a, spart, post, comb)
        torch.cuda.synchronize()

        x_b = x.clone()
        shm.all_gather_sum_mix(part.reshape(-1), x_b, post, comb)
        torch.cuda.synchronize()

        if rank == 0:
            same = torch.equal(out_a, x_b)
            print(f"[case {ci}] rows={rows} D={d}: bitwise_equal={same}", flush=True)
            if not same:
                bad += 1
                diff = (out_a.float() - x_b.float()).abs()
                print(f"  max|diff|={diff.max().item():.6g} at {diff.argmax().item()}", flush=True)

        # copy kernel (all_gather): recv[r] must equal rank r's send, bitwise
        recv = torch.empty(4 * rows * d, dtype=torch.bfloat16, device="cuda")
        shm.all_gather(part.reshape(-1), recv)
        torch.cuda.synchronize()
        expected = torch.cat([
            torch.randn(rows, d, generator=torch.Generator(device="cuda").manual_seed(777 + rr * 100 + ci),
                        device="cuda").to(torch.bfloat16) for rr in range(4)])
        if rank == 0:
            same = torch.equal(recv.view(4, rows, d), expected.view(4, rows, d))
            print(f"[case {ci}] copy-path bitwise_equal={same}", flush=True)
            if not same:
                bad += 1

        store.set(f"done_{ci}_r{rank}", "1")
        store.wait([f"done_{ci}_r{r}" for r in range(4)])
    shm.close()
    if rank == 0:
        print("RESULT", "ALL_BITWISE_EQUAL" if bad == 0 else f"{bad}_CASES_MISMATCH", flush=True)
    return bad


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
    for r, p in enumerate(procs):
        code = p.wait()
        if code:
            print(f"[parent] rank {r} exited {code}", flush=True)
            rc = rc or code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
