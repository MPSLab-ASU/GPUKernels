// flashattention_test.cu
// Host-side test for flashattention kernel.
// Compile with: nvcc -O3 -arch=sm_86 -use_fast_math -Xptxas -dlcm=ca flashattention_kernel.cu flashattention_test.cu -o flashattn_test
#include "flashattention_kernel.cuh"
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)
#endif

// CPU reference attention (FP32)
void cpu_attention_ref(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                       std::vector<float>& O, int B, int H, int N, int D, bool causal) {
    int head_stride = N * D;
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        const float* Q_bh = Q.data() + (b*H + h) * head_stride;
        const float* K_bh = K.data() + (b*H + h) * head_stride;
        const float* V_bh = V.data() + (b*H + h) * head_stride;
        float* O_bh = O.data() + (b*H + h) * head_stride;
        float scale = 1.f / sqrtf((float)D);
        for (int i = 0; i < N; ++i) {
            // compute scores
            std::vector<float> scores(N);
            float m = -INFINITY;
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) { scores[j] = -INFINITY; continue; }
                float dot = 0.f;
                for (int d = 0; d < D; ++d) dot += Q_bh[i*D + d] * K_bh[j*D + d];
                scores[j] = dot * scale;
                m = fmaxf(m, scores[j]);
            }
            float l = 0.f;
            for (int j = 0; j < N; ++j) { if (scores[j] == -INFINITY) continue; float p = expf(scores[j] - m); scores[j]=p; l += p; }
            for (int d = 0; d < D; ++d) {
                float acc = 0.f;
                for (int j = 0; j < N; ++j) if (scores[j] > 0.f) acc += scores[j] * V_bh[j*D + d];
                O_bh[i*D + d] = acc / l;
            }
        }
    }
}

int main() {
    int B = 1, H = 1, N = 64, D = 32; // small test
    size_t sz = (size_t)B*H*N*D;

    std::vector<float> hQf(sz), hKf(sz), hVf(sz), hOf(sz), hOf_ref(sz);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < sz; ++i) { hQf[i] = dist(rng); hKf[i] = dist(rng); hVf[i] = dist(rng); }

    // Convert to half
    std::vector<__half> hQ(sz), hK(sz), hV(sz), hO(sz);
    for (size_t i = 0; i < sz; ++i) { hQ[i] = __float2half(hQf[i]); hK[i] = __float2half(hKf[i]); hV[i] = __float2half(hVf[i]); }

    __half *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, sz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dK, sz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dV, sz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dO, sz * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), sz*sizeof(__half), cudaMemcpyHostToDevice));

    // Launch kernel
    launch_flash_attn_fwd(dQ, dK, dV, dO, B, H, N, D, false);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hO.data(), dO, sz*sizeof(__half), cudaMemcpyDeviceToHost));

    // Convert output to float
    for (size_t i = 0; i < sz; ++i) hOf[i] = __half2float(hO[i]);

    // CPU reference
    cpu_attention_ref(hQf, hKf, hVf, hOf_ref, B, H, N, D, false);

    // Compare
    float max_abs_err = 0.f;
    for (size_t i = 0; i < sz; ++i) {
        float a = hOf[i]; float b = hOf_ref[i];
        float e = fabsf(a-b);
        if (e > max_abs_err) max_abs_err = e;
    }
    printf("max_abs_err = %g\n", max_abs_err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
