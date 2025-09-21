#include <cuda_runtime.h>
#include "sgemm.cuh"

__global__ void sgemm_naive(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void optim_sgemm(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int TILE_SIZE = 32;

    int blockM = blockIdx.y * TILE_SIZE;
    int blockN = blockIdx.x * TILE_SIZE;

    int threadM = blockM + threadIdx.y;
    int threadN = blockN + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE)
    {
        int k_end = min(k + TILE_SIZE, K);

        if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE)
        {
            int globalRow = blockM + threadIdx.y;
            int globalCol = k + threadIdx.x;

            if (globalRow < M && globalCol < K)
            {
                As[threadIdx.y][threadIdx.x] = A[globalRow * K + globalCol];
            }
            else
            {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }

            globalRow = k + threadIdx.y;
            globalCol = blockN + threadIdx.x;

            if (globalRow < K && globalCol < N)
            {
                Bs[threadIdx.y][threadIdx.x] = B[globalRow * N + globalCol];
            }
            else
            {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();

        if (threadM < M && threadN < N)
        {
            for (int kk = k; kk < k_end; ++kk)
            {
                int localK = kk - k;
                float a = As[threadM - blockM][localK];
                float b = Bs[localK][threadN - blockN];
                sum += a * b;
            }
        }

        __syncthreads();
    }

    if (threadM < M && threadN < N)
    {
        C[threadM * N + threadN] = alpha * sum + beta * C[threadM * N + threadN];
    }
}

void sgemm_cuda(
    const float *A, const float *B, float *C,
    int M, int N, int K,
    float alpha, float beta,
    bool use_optimized)
{
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);

    if (use_optimized)
    {
        dim3 blockSize(32, 32);
        dim3 gridSize((N + 31) / 32, (M + 31) / 32);
        optim_sgemm<<<gridSize, blockSize>>>(
            d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    else
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
        sgemm_naive<<<gridSize, blockSize>>>(
            d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}