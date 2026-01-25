
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__
void add(std::size_t N, float* x, float* y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (std::size_t i = index; i < N; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    float *x, *y;

    // Allocate memory for the GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (std::size_t i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add <<<1, 64 >>> (N, x, y);

    // Prevents race conditions
    cudaDeviceSynchronize();

    // Error checking
    float maxError = 0.0f;
    for (std::size_t i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Free Memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}