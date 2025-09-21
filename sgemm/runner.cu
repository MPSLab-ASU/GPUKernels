#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "sgemm.cuh"

void CudaDeviceInfo()
{
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    totalGlobalMem: %zuMB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
           deviceId, props.name, props.major, props.minor,
           props.totalGlobalMem / 1024 / 1024,
           props.multiProcessorCount, props.warpSize);
}

void initialize_matrix(float *matrix, int size, float min_val, float max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < size; ++i)
    {
        matrix[i] = dis(gen);
    }
}

int main()
{

    CudaDeviceInfo();

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::cout << "\nMatrix dimensions: " << M << "x" << K << " * " << K << "x" << N << " = " << M << "x" << N << std::endl;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_basic(M * N, 0.0f);
    std::vector<float> C_optimized(M * N, 0.0f);

    initialize_matrix(A.data(), M * K, 0.0f, 1.0f);
    initialize_matrix(B.data(), K * N, 0.0f, 1.0f);

    const int num_iterations = 10;
    long long total_flops = 2LL * M * N * K;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        sgemm_cuda(A.data(), B.data(), C_basic.data(), M, N, K, alpha, beta, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto basic_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        sgemm_cuda(A.data(), B.data(), C_optimized.data(), M, N, K, alpha, beta, true);
    }
    end = std::chrono::high_resolution_clock::now();
    auto optimized_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_basic_time = static_cast<double>(basic_time.count()) / num_iterations;
    double avg_optimized_time = static_cast<double>(optimized_time.count()) / num_iterations;

    double basic_flops = total_flops / (avg_basic_time / 1000000.0);
    double optimized_flops = total_flops / (avg_optimized_time / 1000000.0);

    std::cout << "\nResults (averaged over " << num_iterations << " iterations):" << std::endl;
    std::cout << "Basic kernel:     " << avg_basic_time << " μs, " << basic_flops / 1e9 << " GFLOPS" << std::endl;
    std::cout << "Warp-tiled:       " << avg_optimized_time << " μs, " << optimized_flops / 1e9 << " GFLOPS" << std::endl;

    if (avg_optimized_time > 0)
    {
        double speedup = avg_basic_time / avg_optimized_time;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }

    bool results_match = true;
    const float tolerance = 1e-3f;
    int mismatch_count = 0;
    for (int i = 0; i < M * N; ++i)
    {
        if (abs(C_basic[i] - C_optimized[i]) > tolerance)
        {
            results_match = false;
            mismatch_count++;
            if (mismatch_count <= 5)
            {
                std::cout << "Mismatch at " << i << ": basic=" << C_basic[i]
                          << ", optimized=" << C_optimized[i]
                          << ", diff=" << abs(C_basic[i] - C_optimized[i]) << std::endl;
            }
        }
    }
    if (mismatch_count > 5)
    {
        std::cout << "Total mismatches: " << mismatch_count << std::endl;
    }

    std::cout << "Results match: " << (results_match ? "PASS" : "FAIL") << std::endl;
}