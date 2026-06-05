#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ─────────────────────────────────────────────
// THE KERNEL — runs on the GPU
// __global__ = "launch from CPU, run on GPU"
// Every thread runs this function once,
// but each thread handles a different index i.
// ─────────────────────────────────────────────
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Each thread computes its own global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: the grid may be slightly larger than n
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// ─────────────────────────────────────────────
// HELPER — check CUDA errors
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    // ── 1. Setup ──────────────────────────────
    int n = 1 << 24;              // 16M elements (~64 MB per array)
    size_t bytes = n * sizeof(float);

    // ── 2. Allocate on CPU (host) ─────────────
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }

    // ── 3. Allocate on GPU (device) ───────────
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ── 4. Copy CPU → GPU ─────────────────────
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ── 5. Launch the kernel ──────────────────
    // 256 threads per block is a common default.
    // blocks = ceil(n / 256) so every element is covered.
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());   // catch launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for GPU to finish

    // ── 6. Copy GPU → CPU ─────────────────────
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ── 7. Verify ─────────────────────────────
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];   // always = n
        if (h_c[i] != expected) errors++;
    }
    printf("n = %d\n", n);
    printf("errors = %d\n", errors);
    printf("c[0]   = %.0f (expected %.0f)\n", h_c[0],   (float)n);
    printf("c[100] = %.0f (expected %.0f)\n", h_c[100], (float)n);

    // ── 8. Free memory ────────────────────────
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
