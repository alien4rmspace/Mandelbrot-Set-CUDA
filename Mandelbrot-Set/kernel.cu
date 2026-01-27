#include "kernel_api.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mandelbrotItersKernel(unsigned short* iters, MandelbrotParams p)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    // Map pixel -> complex plane
    double u = (x + 0.5f) / p.width;   // [0,1]
    double v = 1.0f - (float)y / (p.height - 1); // Y is flipped due to SFML. Top screen max imaginary.

    double cx = p.centerX + (u - 0.5f) * p.sizeX;
    double cy = p.centerY + (v - 0.5f) * p.sizeY;

    double zx = 0.0f, zy = 0.0f;
    int iter = 0;

    for (; iter < p.maxIter; ++iter) {
        double zx2 = zx * zx - zy * zy + cx;
        double zy2 = 2.0f * zx * zy + cy;
        zx = zx2;
        zy = zy2;

        if (zx * zx + zy * zy > 4.0f) break;
    }

    iters[y * p.width + x] = (unsigned short)iter;  // y * p.width basically a stride.
}

__global__ void addKernel(int N, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        y[i] = x[i] + y[i];
    }
}

extern "C" void launchMandelbrotIters(unsigned short* d_iters, const MandelbrotParams* params) {
    dim3 block(16, 16); // 256 threads per block
    dim3 grid(
        (params->width + block.x - 1) / block.x,    // (1080 + 16 - 1) / 16 = 68
        (params->height + block.y - 1) / block.y
    );

    mandelbrotItersKernel <<<grid, block >> > (d_iters, *params);
}

extern "C" void launchAdd(int N, float* x, float* y) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addKernel <<<numBlocks, blockSize >>> (N, x, y);
}
