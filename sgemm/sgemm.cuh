#ifndef SGEMM_H
#define SGEMM_H

#include <cuda_runtime.h>

void sgemm_cuda(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    bool use_optimized);

void initialize_matrix(float *matrix, int size, float min_val = 0.0f, float max_val = 1.0f);
bool verify_results(const float *C1, const float *C2, int size, float tolerance = 1e-5f);

#endif
