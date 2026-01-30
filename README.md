# Mandelbrot Set (GPU-Accelerated)

High-performance Mandelbrot set renderer with smooth deep-zoom support.

▶️ **15-second demo (real-time zoom):**  


https://github.com/user-attachments/assets/81255ab7-adfd-4d16-9171-d43a5e2a0b2b


## Technical Highlights
- GPU-parallel computation (CUDA)
- Smooth continuous zoom without frame stutter
- High-precision arithmetic for deep zooms
- Performance profiled and optimized using NVIDIA Nsight Systems

## Profiling & Performance Analysis

**Figure 1:** Nsight Systems timeline showing repeated executions of
`ComplexPlane::updateRender()` caused by failing to transition from
`CALCULATING` to `DISPLAYING`. The missing state update resulted in
unnecessary recomputation each render loop iteration.
<img width="1028" height="385" alt="image" src="https://github.com/user-attachments/assets/48e0998c-3978-4408-8612-f2508255e8c9" />

**Figure 2:** Nsight Systems timeline after correcting the state
transition (`m_state = DISPLAYING`). `updateRender()` now executes only
once per interaction instead of every frame, confirming the issue was
caused by incorrect state management.
<img width="1033" height="363" alt="image" src="https://github.com/user-attachments/assets/f1823472-788a-46b5-83c0-f1fc12163630" />

**Figure 3:** Nsight Systems NVTX timing for ComplexPlane::updateRender() executed on the CPU. The function execution time is ~374 ms for a single full-frame Mandelbrot computation.
<img width="1034" height="359" alt="Screenshot 2026-01-30 125547" src="https://github.com/user-attachments/assets/ed27cf3b-5639-40ef-9ff9-f0db0c0202a7" />

**Figure 4:** Nsight Systems timing of ComplexPlane::updateRender() executed on the GPU. The total per-frame execution time is ~1.0 ms, with the Mandelbrot CUDA kernel accounting
for ~0.87 ms. The remaining ~0.13 ms is spent in a single-threaded CPU loop responsible for post-processing and vertex/color updates, indicating potential for further
optimization by reducing CPU-side work.
<img width="1032" height="378" alt="Screenshot 2026-01-30 125614" src="https://github.com/user-attachments/assets/4163ad5c-1822-421c-b69b-7a25bd018fe9" />

## Performance Summary
The measurements in **Figures 3** and **4** are summarized below.
| Implementation | Time per frame |
|---------------|---------------|
| CPU (single-threaded) | ~374 ms |
| GPU (CUDA) | ~1.0 ms |
| Speedup | ~370× |

These results demonstrate the effectiveness of GPU parallelization for
parallel workloads such as fractal rendering.
