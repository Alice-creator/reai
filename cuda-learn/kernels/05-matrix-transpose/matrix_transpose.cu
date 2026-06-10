#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 2D kernel: mỗi thread xử lý 1 phần tử (row, col)
// in[row][col]  →  out[col][row]  (transpose)
__global__ void matrix_transpose(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int new_row  = col;   // row trong output  = original col
    int new_col  = row;   // col trong output  = original row
    int new_cols = rows;  // width của output  = original rows

    // Bounds guard: check CẢ HAI chiều
    if (row >= rows || col >= cols) return;

    out[new_row * new_cols + new_col] = in[row * cols + col];
}

int main() {
    int rows = 1024, cols = 2048;
    size_t in_bytes  = rows * cols * sizeof(float);
    size_t out_bytes = cols * rows * sizeof(float);  // same size, different shape

    float* h_in  = (float*)malloc(in_bytes);
    float* h_out = (float*)malloc(out_bytes);

    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));

    // 16×16 block = 256 threads — tiêu chuẩn cho matrix kernels
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    matrix_transpose<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
