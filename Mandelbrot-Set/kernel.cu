#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernel_api.h"

__global__ void addKernel(int N, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        y[i] = x[i] + y[i];
    }
}

extern "C" void launchAdd(int N, float* x, float* y) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addKernel <<<numBlocks, blockSize >>> (N, x, y);
}
