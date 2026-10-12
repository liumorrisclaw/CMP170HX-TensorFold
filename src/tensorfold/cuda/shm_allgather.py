"""Host-staged all-gather for TP ranks without peer access (sm_80 CMP 170HX).

TensorFold's all-gather (cuda/comm.py) re-expressed over one POSIX shm segment
mmap'd by every rank: each GPU holds a device pointer to the same physical pages
(cudaHostRegister Mapped|Portable), so a kernel can stage its slice of the
payload into its own slot, announce an arrival word, and read the peers' slots
straight out of host memory.

Messages above ``FALLBACK_BYTES`` (large prefill payloads) stay on NCCL; only
the one-shot path lives here.

Capturable: the generation counter lives in a device tensor the kernel itself
advances, never in an argument, so a captured graph replays correctly. The base
device address is fixed at registration time, so passing it as a kernel argument
is graph-safe too. Every wait is bounded by clock64(); on timeout the rank marks
a 64-bit failure word and leaves its output untouched so validation shouts.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import time

import torch

MAX_RANKS = 4
MAX_BLOCKS = 4
HEADER_BYTES = 256 * 1024
SLOT_BYTES = 1 << 20            # per rank per parity; covers R<=64 decode windows
FALLBACK_BYTES = SLOT_BYTES     # above this, the caller keeps NCCL
_SHM_BYTES = HEADER_BYTES + 2 * MAX_RANKS * SLOT_BYTES

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cstdint>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e));   \
  } while (0)

namespace tfshm {

constexpr int WORLD = 4;                   // every kernel here assumes a TP4 group
constexpr int STAGES = 2;                  // parity staging areas: one call overlaps the previous
constexpr uint32_t GAVE_UP = 0x80000000u;  // a timed-out rank tags its arrival word with this

using SysWord32 = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
using SysWord64 = cuda::atomic_ref<unsigned long long, cuda::thread_scope_system>;

// Shared-page layout: a small control region (per-(stage, block, rank) arrival
// words plus one sticky 64-bit failure word per rank), then STAGES rank-ordered
// staging areas of SLOT_BYTES each. Fixed at registration; kernels derive every
// offset from `base`.
struct alignas(256) Layout {
  uint32_t arrival[STAGES][4][WORLD];      // [stage][block][rank] = generation tag
  unsigned long long failure[WORLD];       // [rank] = (gen << 1) | 1, written once
};

template <typename T, int N>
struct __align__(sizeof(T) * N) Run { T d[N]; };

__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint8_t* stage_area(uint8_t* base, int stage, int64_t slot_bytes) {
  return base + sizeof(Layout) + static_cast<int64_t>(stage) * WORLD * slot_bytes;
}

// The staging writes must be visible before the tag lands: every thread drains its
// own writes with a system fence, then thread 0 alone publishes the generation with
// one release store (one store per block, ordered after the whole CTA's staging).
__device__ __forceinline__ void announce(uint32_t* word, uint32_t gen) {
  __threadfence_system();
  if (threadIdx.x == 0)
    SysWord32(*word).store(gen, cuda::memory_order_release);
}

// Parallel wait: one thread per peer spins on that peer's arrival word (parallel host
// reads keep detection at one PCIe round), the rest of the block catches up at the
// barrier. A rank that blows its budget tags its own word with GAVE_UP, records the
// sticky failure word and gives up -- its output is left untouched so validation
// shouts, and its generation still advances so the next call starts clean.
__device__ __forceinline__ bool peers_ready(Layout* lay, int stage, int block, int rank,
                                            uint32_t gen, unsigned long long budget,
                                            int* gave_up) {
  const int tid = threadIdx.x;
  if (tid == 0) *gave_up = 0;
  __syncthreads();
  if (tid < WORLD && tid != rank) {
    uint32_t* word = &lay->arrival[stage][block][tid];
    const long long t0 = clock64();
    while (SysWord32(*word).load(cuda::memory_order_acquire) != gen) {
      if ((unsigned long long)(clock64() - t0) > budget) {
        SysWord32(lay->arrival[stage][block][rank]).store(gen | GAVE_UP, cuda::memory_order_release);
        SysWord64(lay->failure[rank]).store(((unsigned long long)gen << 1) | 1ULL,
                                            cuda::memory_order_relaxed);
        *gave_up = 1;
        break;
      }
#if __CUDA_ARCH__ >= 700
      __nanosleep(32);
#endif
    }
  }
  __syncthreads();
  return *gave_up != 0;
}

template <typename T, int N>
__device__ __forceinline__ const Run<T, N>* staged(uint8_t* area, int rank, int64_t slot_bytes) {
  return reinterpret_cast<const Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);
}

// ---- one-shot all-gather: rank r's slice lands at [r * nunits]; pure copies ----
template <typename T, int N, int NB>
__global__ void __launch_bounds__(1024, 1)
gather_copies(const Run<T, N>* __restrict__ slice,
              Run<T, N>* __restrict__ out,
              uint8_t* __restrict__ base,
              uint32_t* __restrict__ seq,
              const int rank,
              const int64_t nunits,
              const int64_t slot_bytes,
              const unsigned long long budget) {
  Layout* lay = reinterpret_cast<Layout*>(base);
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int stage = static_cast<int>(gen & 1u);
  uint8_t* const area = stage_area(base, stage, slot_bytes);
  Run<T, N>* const mine = reinterpret_cast<Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);

  const int64_t span = (nunits + NB - 1) / NB;
  const int64_t first = static_cast<int64_t>(b) * span;
  const int64_t last = (first + span) < nunits ? (first + span) : nunits;

  __shared__ int gave_up;

  for (int64_t i = first + tid; i < last; i += nt) mine[i] = slice[i];
  announce(&lay->arrival[stage][b][rank], gen);

  if (peers_ready(lay, stage, b, rank, gen, budget, &gave_up)) {
    if (tid == 0) seq[b] = gen;
    return;
  }

  for (int64_t i = first + tid; i < last; i += nt) {
#pragma unroll
    for (int r = 0; r < WORLD; ++r) {
      const Run<T, N>* from = (r == rank) ? slice : staged<T, N>(area, r, slot_bytes);
      out[static_cast<int64_t>(r) * nunits + i] = from[i];
    }
  }

  if (tid == 0) seq[b] = gen;
}

// ---- one-shot all-gather with the rank-order sum folded in: out = bf16(sum in rank order) ----
// Same values, same order, one bf16 rounding as all_gather plus the consumer's fp32
// sum pass -- replies stay bit-identical to the unfused path.
template <typename T, int N, int NB>
__global__ void __launch_bounds__(1024, 1)
gather_sum(const Run<T, N>* __restrict__ slice,
           Run<T, N>* __restrict__ out,
           uint8_t* __restrict__ base,
           uint32_t* __restrict__ seq,
           const int rank,
           const int64_t nunits,
           const int64_t slot_bytes,
           const unsigned long long budget) {
  Layout* lay = reinterpret_cast<Layout*>(base);
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int stage = static_cast<int>(gen & 1u);
  uint8_t* const area = stage_area(base, stage, slot_bytes);
  Run<T, N>* const mine = reinterpret_cast<Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);

  const int64_t span = (nunits + NB - 1) / NB;
  const int64_t first = static_cast<int64_t>(b) * span;
  const int64_t last = (first + span) < nunits ? (first + span) : nunits;

  __shared__ int gave_up;

  for (int64_t i = first + tid; i < last; i += nt) mine[i] = slice[i];
  announce(&lay->arrival[stage][b][rank], gen);

  if (peers_ready(lay, stage, b, rank, gen, budget, &gave_up)) {
    if (tid == 0) seq[b] = gen;
    return;
  }

  for (int64_t i = first + tid; i < last; i += nt) {
    const Run<T, N>* first_in = (rank == 0) ? slice : staged<T, N>(area, 0, slot_bytes);
    float part[N];
#pragma unroll
    for (int j = 0; j < N; ++j) part[j] = widen(first_in[i].d[j]);
#pragma unroll
    for (int r = 1; r < WORLD; ++r) {
      const Run<T, N>* from = (rank == r) ? slice : staged<T, N>(area, r, slot_bytes);
#pragma unroll
      for (int j = 0; j < N; ++j) part[j] += widen(from[i].d[j]);
    }
    Run<T, N> summed;
#pragma unroll
    for (int j = 0; j < N; ++j) summed.d[j] = __float2bfloat16(part[j]);
    out[i] = summed;
  }

  if (tid == 0) seq[b] = gen;
}

// ---- all-gather + rank-order sum + the hc stream mix in one kernel: x updates in place ----
// Per unit i (row r = i/dunits, column vec d4 = i%dunits): the summed partial rounds
// once to bf16 (the branch), then stream s takes v_s = ps_s * branch + mix_s with
//   mix_s = fma(x0,c0, x1*c1) -> fma(x2,c2, ->) -> fma(x3,c3, ->)
// matching the Triton _hc_mix contraction bit for bit; the four patched streams land
// in x itself.
template <typename T, int N, int NB>
__global__ void __launch_bounds__(1024, 1)
gather_sum_streammix(const Run<T, N>* __restrict__ slice,
                     Run<T, N>* __restrict__ x,
                     const float* __restrict__ post,
                     const float* __restrict__ comb,
                     uint8_t* __restrict__ base,
                     uint32_t* __restrict__ seq,
                     const int rank,
                     const int64_t nunits,
                     const int64_t dunits,
                     const int64_t slot_bytes,
                     const unsigned long long budget) {
  Layout* lay = reinterpret_cast<Layout*>(base);
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint32_t gen = seq[b] + 1u;
  const int stage = static_cast<int>(gen & 1u);
  uint8_t* const area = stage_area(base, stage, slot_bytes);
  Run<T, N>* const mine = reinterpret_cast<Run<T, N>*>(area + static_cast<int64_t>(rank) * slot_bytes);

  const int64_t span = (nunits + NB - 1) / NB;
  const int64_t first = static_cast<int64_t>(b) * span;
  const int64_t last = (first + span) < nunits ? (first + span) : nunits;

  __shared__ int gave_up;

  for (int64_t i = first + tid; i < last; i += nt) mine[i] = slice[i];
  announce(&lay->arrival[stage][b][rank], gen);

  if (peers_ready(lay, stage, b, rank, gen, budget, &gave_up)) {
    if (tid == 0) seq[b] = gen;
    return;
  }

  // uniform pass count: the in-loop barrier must see every thread of the block
  const int passes = static_cast<int>((last - first + nt - 1) / nt);
  for (int k = 0; k < passes; ++k) {
    const int64_t i = first + static_cast<int64_t>(k) * nt + tid;
    const bool on = i < last;
    const int64_t r = on ? i / dunits : 0;
    const int64_t d4 = on ? i - r * dunits : 0;
    float branch[N];
    Run<T, N> stream_in[4];
    if (on) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        const Run<T, N>* head = (rank == 0) ? slice : staged<T, N>(area, 0, slot_bytes);
        float acc = widen(head[i].d[j]);
#pragma unroll
        for (int rr = 1; rr < WORLD; ++rr) {
          const Run<T, N>* from = (rank == rr) ? slice : staged<T, N>(area, rr, slot_bytes);
          acc += widen(from[i].d[j]);
        }
        branch[j] = widen(__float2bfloat16(acc));
      }
#pragma unroll
      for (int g = 0; g < 4; ++g) stream_in[g] = x[(r * 4 + g) * dunits + d4];
    }
    __syncthreads();   // the reads above own every address the writes below touch
    if (on) {
      const float* po = post + r * 4;
      const float* co = comb + r * 16;
      float xs[4][N];
#pragma unroll
      for (int g = 0; g < 4; ++g)
#pragma unroll
        for (int j = 0; j < N; ++j) xs[g][j] = widen(stream_in[g].d[j]);
      Run<T, N> patched[4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const float ps = po[s];
#pragma unroll
        for (int j = 0; j < N; ++j) {
          float mixed = fmaf(xs[0][j], co[0 * 4 + s], xs[1][j] * co[1 * 4 + s]);
          mixed = fmaf(xs[2][j], co[2 * 4 + s], mixed);
          mixed = fmaf(xs[3][j], co[3 * 4 + s], mixed);
          patched[s].d[j] = __float2bfloat16(fmaf(branch[j], ps, mixed));
        }
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) x[(r * 4 + s) * dunits + d4] = patched[s];
    }
  }

  if (tid == 0) seq[b] = gen;
}

int64_t failure_word(int64_t base_ptr) {
  const Layout* lay = reinterpret_cast<const Layout*>(base_ptr);
  unsigned long long worst = 0;
  for (int r = 0; r < WORLD; ++r) worst |= lay->failure[r];
  return static_cast<int64_t>(worst);
}

std::tuple<int64_t, int64_t> host_register(int64_t ptr, int64_t nbytes) {
  cudaError_t e = cudaHostRegister(reinterpret_cast<void*>(ptr),
                                   static_cast<size_t>(nbytes),
                                   cudaHostRegisterMapped | cudaHostRegisterPortable);
  TORCH_CHECK(e == cudaSuccess, "cudaHostRegister: ", cudaGetErrorString(e),
              " (if a previous run left pages registered, rm the shm file and retry)");
  void* dev = nullptr;
  e = cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(ptr), 0);
  TORCH_CHECK(e == cudaSuccess, "cudaHostGetDevicePointer: ", cudaGetErrorString(e));
  return {reinterpret_cast<int64_t>(dev), ptr};
}

void host_unregister(int64_t ptr) { cudaHostUnregister(reinterpret_cast<void*>(ptr)); }

}  // namespace tfshm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("host_register", &tfshm::host_register);
  m.def("host_unregister", &tfshm::host_unregister);
  m.def("failure_word", &tfshm::failure_word);
  m.def("run_sum_bf16", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                           int64_t rank, int64_t nunits, int64_t slot_bytes,
                           int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<__nv_bfloat16, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<__nv_bfloat16, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_sum<__nv_bfloat16, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_sum<__nv_bfloat16, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_sum<__nv_bfloat16, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_sum_mix_bf16", [](int64_t inp, int64_t x, int64_t post, int64_t comb,
                               int64_t base, int64_t seq, int64_t rank, int64_t nunits,
                               int64_t dunits, int64_t slot_bytes, int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<__nv_bfloat16, 4>*>(inp);
    auto* x_v = reinterpret_cast<Run<__nv_bfloat16, 4>*>(x);
    auto* po = reinterpret_cast<const float*>(post);
    auto* co = reinterpret_cast<const float*>(comb);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_sum_streammix<__nv_bfloat16, 4, 1><<<1, 1024, 0, stream>>>(in_v, x_v, po, co, bb, sq, rk, nunits, dunits, slot_bytes, bud);
    else if (blocks == 2) gather_sum_streammix<__nv_bfloat16, 4, 2><<<2, 1024, 0, stream>>>(in_v, x_v, po, co, bb, sq, rk, nunits, dunits, slot_bytes, bud);
    else gather_sum_streammix<__nv_bfloat16, 4, 4><<<4, 1024, 0, stream>>>(in_v, x_v, po, co, bb, sq, rk, nunits, dunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_f32", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                      int64_t rank, int64_t nunits, int64_t slot_bytes,
                      int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<float, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<float, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<float, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<float, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<float, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_bf16", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                       int64_t rank, int64_t nunits, int64_t slot_bytes,
                       int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<__nv_bfloat16, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<__nv_bfloat16, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<__nv_bfloat16, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<__nv_bfloat16, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<__nv_bfloat16, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
  m.def("run_i32", [](int64_t inp, int64_t out, int64_t base, int64_t seq,
                      int64_t rank, int64_t nunits, int64_t slot_bytes,
                      int64_t spin, int64_t blocks) {
    using namespace tfshm;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto* in_v = reinterpret_cast<const Run<int, 4>*>(inp);
    auto* out_v = reinterpret_cast<Run<int, 4>*>(out);
    auto* bb = reinterpret_cast<uint8_t*>(base);
    auto* sq = reinterpret_cast<uint32_t*>(seq);
    const int rk = static_cast<int>(rank);
    const auto bud = static_cast<unsigned long long>(spin);
    if (blocks == 1) gather_copies<int, 4, 1><<<1, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else if (blocks == 2) gather_copies<int, 4, 2><<<2, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    else gather_copies<int, 4, 4><<<4, 1024, 0, stream>>>(in_v, out_v, bb, sq, rk, nunits, slot_bytes, bud);
    CUDA_CHECK(cudaGetLastError());
  });
}
"""


def _build_module():
    from pathlib import Path

    from tensorfold.cuda.build import load

    src = Path(__file__).parent / "_shm_allgather.cu"
    if not src.exists() or src.stat().st_size != len(CUDA_SRC):
        src.write_text(CUDA_SRC)
    return load(name="tensorfold_shm_allgather_v3", sources=[str(src)],
                extra_cuda_cflags=["-O3", "-std=c++17"], verbose=False)


class ShmAllGather:
    """One shm segment per TP group; the same all_gather(send, recv) contract as NCCL.all_gather."""

    def __init__(self, rank: int, world: int, store, *, blocks: int = MAX_BLOCKS) -> None:
        if world != 4:
            raise ValueError(f"shm all-gather port targets world=4, got {world}")
        self.rank, self.world, self.store = rank, world, store
        self.blocks = blocks
        self.name = f"/dev/shm/tf_ag_{store.port}"
        if rank == 0:
            fd = os.open(self.name, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, _SHM_BYTES)
            os.close(fd)
            store.set("tf_ag_created", "1")
        else:
            for _ in range(600):
                try:
                    if store.get("tf_ag_created") == "1":
                        break
                except Exception:
                    time.sleep(0.5)
        fd = os.open(self.name, os.O_RDWR)
        self._map = mmap.mmap(fd, _SHM_BYTES)
        os.close(fd)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        torch.cuda.init()
        self.ext = _build_module()
        self.dev_base, self.host_addr = self.ext.host_register(addr, _SHM_BYTES)
        self.seq = torch.zeros(blocks, dtype=torch.int32, device="cuda")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order. Fallback is the caller's job."""
        nbytes = send.numel() * send.element_size()
        if send.dtype not in (torch.float32, torch.bfloat16, torch.int32):
            raise ValueError(f"shm all-gather dtype {send.dtype}")
        vec_bytes = 8 if send.dtype is torch.bfloat16 else 16   # Run<T, 4> width
        if nbytes > FALLBACK_BYTES or nbytes % vec_bytes or nbytes == 0:
            raise ValueError(f"shm all-gather size {nbytes}")
        if not send.is_contiguous() or not recv.is_contiguous():
            raise ValueError("shm all-gather needs contiguous tensors")
        fn = {torch.float32: self.ext.run_f32,
              torch.bfloat16: self.ext.run_bf16,
              torch.int32: self.ext.run_i32}[send.dtype]
        fn(send.data_ptr(), recv.data_ptr(), self.dev_base, self.seq.data_ptr(),
           self.rank, nbytes // vec_bytes, SLOT_BYTES, 1 << 30, self.blocks)

    def all_gather_sum(self, send: torch.Tensor, out: torch.Tensor) -> None:
        """out [n] bf16 <- the rank-order fp32 sum of every rank's bf16 send [n], in one kernel.

        Identical values to all_gather + a rank-order fp32 sum + one bf16 round -- the
        consumer's separate sum pass folds into the gather. world=4, bf16 only."""
        nbytes = send.numel() * send.element_size()
        if send.dtype is not torch.bfloat16:
            raise ValueError("all_gather_sum: bf16 only")
        if nbytes > FALLBACK_BYTES or nbytes % 8 or nbytes == 0:
            raise ValueError(f"all_gather_sum size {nbytes}")
        if not send.is_contiguous() or not out.is_contiguous():
            raise ValueError("all_gather_sum needs contiguous tensors")
        self.ext.run_sum_bf16(send.data_ptr(), out.data_ptr(), self.dev_base, self.seq.data_ptr(),
                              self.rank, nbytes // 8, SLOT_BYTES, 1 << 30, self.blocks)

    def all_gather_sum_mix(self, send: torch.Tensor, x: torch.Tensor, post: torch.Tensor,
                           comb: torch.Tensor) -> None:
        """Fold the hc stream mix into the gather: x [rows, 4*D] bf16 is updated in place with
        v_s = ps_s * branch + mix_s, branch being the bf16-rounded rank-order sum of every
        rank's send [rows, D]. Identical bits to all_gather_sum + glue.hc_mix (same fma shape,
        same rounding points); the spart round trip and the extra launch disappear.
        world=4, 4 streams, bf16 send/x, fp32 post/comb."""
        rows, wide = x.shape
        d = send.numel() // rows
        if send.dtype is not torch.bfloat16 or x.dtype is not torch.bfloat16:
            raise ValueError("all_gather_sum_mix: bf16 send/x only")
        if post.dtype is not torch.float32 or comb.dtype is not torch.float32:
            raise ValueError("all_gather_sum_mix: fp32 post/comb only")
        if wide != 4 * d or post.numel() != rows * 4 or comb.numel() != rows * 16:
            raise ValueError(f"all_gather_sum_mix shapes send {tuple(send.shape)} x {tuple(x.shape)}")
        nbytes = send.numel() * send.element_size()
        if nbytes > FALLBACK_BYTES or nbytes % 8 or nbytes == 0 or d % 4:
            raise ValueError(f"all_gather_sum_mix size {nbytes}")
        if not send.is_contiguous() or not x.is_contiguous() or not post.is_contiguous() \
                or not comb.is_contiguous():
            raise ValueError("all_gather_sum_mix needs contiguous tensors")
        self.ext.run_sum_mix_bf16(send.data_ptr(), x.data_ptr(), post.data_ptr(), comb.data_ptr(),
                                  self.dev_base, self.seq.data_ptr(), self.rank, nbytes // 8,
                                  d // 4, SLOT_BYTES, 1 << 30, self.blocks)

    def ok(self) -> bool:
        """True unless some rank recorded a wait timeout (checked outside graphs)."""
        return int(self.ext.failure_word(self.dev_base)) == 0

    def close(self) -> None:
        try:
            self.ext.host_unregister(self.host_addr)
        except Exception:
            pass
        try:
            self._map.close()
            if self.rank == 0:
                os.unlink(self.name)
        except Exception:
            pass
