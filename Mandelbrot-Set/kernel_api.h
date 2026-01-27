#pragma once

#include "mandelbrot_params.h"

#ifdef __cplusplus
extern "C" {
#endif

	void launchAdd(int N, float* x, float* y);

	void launchMandelbrotIters(unsigned short* d_iters, const MandelbrotParams* params);

#ifdef __cplusplus
}
#endif