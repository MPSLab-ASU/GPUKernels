// flashattention_kernel.cu
// Kernel implementation for FlashAttention-like forward pass.

#include "flashattention_kernel.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)
#endif

// -------------------- Tunables --------------------
#ifndef TILE_Q
#define TILE_Q 64     // queries per block (rows)
#endif
#ifndef TILE_K
#define TILE_K 64     // keys per tile (cols) processed per iteration
#endif
#ifndef THREADS
#define THREADS 256   // threads per block (multiple of 32)
#endif

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline __device__ __half2 h2ld(const __half* p) {
    return *reinterpret_cast<const __half2*>(p);
}

extern "C" __global__ void flash_attn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int B, int H, int N, int D,
    float scale,
    int causal
) {
    extern __shared__ __half smem[];
    __half* Ks = smem;
    __half* Vs = Ks + (size_t)TILE_K * D;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_tile_start = blockIdx.x * TILE_Q;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    size_t head_stride = (size_t)N * D;
    size_t bh_offset = ((size_t)b * H + h) * head_stride;
    const __half* Q_bh = Q + bh_offset;
    const __half* K_bh = K + bh_offset;
    const __half* V_bh = V + bh_offset;
    __half* O_bh = O + bh_offset;

    int vec = (D % 2 == 0) ? 2 : 1;
    const int WARPS = THREADS / 32;
    const int ROWS_PER_WARP = ceil_div(TILE_Q, WARPS);

    for (int row_iter = 0; row_iter < ROWS_PER_WARP; ++row_iter) {
        int q_row = q_tile_start + warp_id + row_iter * WARPS;
        if (q_row >= N) continue;

        for (int d0 = lane_id * vec; d0 < D; d0 += 32 * vec) {
            float o_acc[2] = {0.f, 0.f};
            float m_i = -INFINITY;
            float l_i = 0.f;


            for (int k_tile = 0; k_tile < N; k_tile += TILE_K) {
                int cols = min(TILE_K, N - k_tile);

                for (int idx = tid; idx < cols * D; idx += THREADS) {
                    int r = idx / D;
                    int c = idx % D;
                    Ks[r * D + c] = K_bh[(size_t)(k_tile + r) * D + c];
                    Vs[r * D + c] = V_bh[(size_t)(k_tile + r) * D + c];
                }
                __syncthreads();

                float local_max = -INFINITY;

                for (int j = 0; j < cols; ++j) {
                    const __half* krow = &Ks[j * D];
                    float dot = 0.f;
                    if (D % 2 == 0) {
                        const __half2* k2 = reinterpret_cast<const __half2*>(krow);
                        const __half2* q2 = reinterpret_cast<const __half2*>(Q_bh + (size_t)q_row * D);
                        for (int t = 0; t < D/2; ++t) {
                            __half2 a = q2[t];
                            __half2 b = k2[t];
                            float a0 = __half2float(__low2half(a));
                            float a1 = __half2float(__high2half(a));
                            float b0 = __half2float(__low2half(b));
                            float b1 = __half2float(__high2half(b));
                            dot += a0 * b0 + a1 * b1;
                        }
                    } else {
                        for (int d = 0; d < D; ++d) {
                            dot += __half2float(Q_bh[(size_t)q_row * D + d]) * __half2float(krow[d]);
                        }
                    }

                    int q_pos = q_row;
                    int k_pos = k_tile + j;
                    if (causal && k_pos > q_pos) {
                        dot = -INFINITY;
                    }

                    float s = dot * scale;
                    local_max = fmaxf(local_max, s);
                }

                float m_new = fmaxf(m_i, local_max);
                float alpha = expf(m_i - m_new);
                l_i *= alpha;
                o_acc[0] *= alpha; o_acc[1] *= alpha;
                m_i = m_new;

                for (int j = 0; j < cols; ++j) {
                    const __half* krow = &Ks[j * D];
                    float dot = 0.f;
                    if (D % 2 == 0) {
                        const __half2* k2 = reinterpret_cast<const __half2*>(krow);
                        const __half2* q2 = reinterpret_cast<const __half2*>(Q_bh + (size_t)q_row * D);
                        for (int t = 0; t < D/2; ++t) {
                            __half2 a = q2[t];
                            __half2 b = k2[t];
                            float a0 = __half2float(__low2half(a));
                            float a1 = __half2float(__high2half(a));
                            float b0 = __half2float(__low2half(b));
                            float b1 = __half2float(__high2half(b));
                            dot += a0 * b0 + a1 * b1;
                        }
                    } else {
                        for (int t = 0; t < D; ++t) {
                            dot += __half2float(Q_bh[(size_t)q_row * D + t]) * __half2float(krow[t]);
                        }
                    }

                    int q_pos = q_row;
                    int k_pos = k_tile + j;
                    if (causal && k_pos > q_pos) {
                        continue;
                    }

                    float s = dot * scale;
                    float p = expf(s - m_i);
                    l_i += p;

                    const __half* vrow = &Vs[j * D + d0];
                    if (vec == 2 && d0 + 1 < D) {
                        __half2 vh2 = h2ld(vrow);
                        float v0 = __half2float(__low2half(vh2));
                        float v1 = __half2float(__high2half(vh2));
                        o_acc[0] += p * v0;
                        o_acc[1] += p * v1;
                    } else {
                        float v0 = __half2float(*vrow);
                        o_acc[0] += p * v0;
                    }
                }

                __syncthreads();
            }

            if (l_i > 0.f) {
                float inv_l = 1.f / l_i;
                __half* o_ptr = O_bh + (size_t)q_row * D + d0;
                if (vec == 2 && d0 + 1 < D) {
                    __half2 out;
                    out = __halves2half2(__float2half(o_acc[0] * inv_l), __float2half(o_acc[1] * inv_l));
                    *reinterpret_cast<__half2*>(o_ptr) = out;
                } else {
                    *o_ptr = __float2half(o_acc[0] * inv_l);
                }
            }
        }
    }
}

extern "C" void launch_flash_attn_fwd(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int B, int H, int N, int D, bool causal
) {
    float scale = 1.f / sqrtf((float)D);
    dim3 grid(ceil_div(N, TILE_Q), H, B);
    dim3 block(THREADS);
    size_t shmem = (size_t)(TILE_K * D + TILE_K * D) * sizeof(__half);
    flash_attn_fwd<<<grid, block, shmem>>>(Q, K, V, O, B, H, N, D, scale, (int)causal);
}
