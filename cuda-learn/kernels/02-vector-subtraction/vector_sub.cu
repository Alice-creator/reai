#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// TODO: viết kernel vector subtraction ở đây
// Công thức: c[i] = a[i] - b[i]
// Gợi ý: copy từ vector_add.cu, chỉ đổi một ký tự
__global__ void vector_sub(const float* a, const float* b, float* c, int n) {
    // TODO
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    int n     = 1 << 24;
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i / 2);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vector_sub<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify: c[i] should equal i - i/2
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] - h_b[i];
        if (h_c[i] != expected) errors++;
    }
    printf("n = %d\n", n);
    printf("errors = %d\n", errors);
    printf("c[0]   = %.0f (expected %.0f)\n", h_c[0],   h_a[0]   - h_b[0]);
    printf("c[100] = %.0f (expected %.0f)\n", h_c[100], h_a[100] - h_b[100]);
    printf("c[999] = %.0f (expected %.0f)\n", h_c[999], h_a[999] - h_b[999]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
