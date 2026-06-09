#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

__global__ void matrix_transpose(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    out[col * rows + row] = in[row * cols + col];
}

// Helper: chạy transpose, trả về output trên CPU
std::vector<float> run_transpose(const std::vector<float>& input, int rows, int cols) {
    size_t in_bytes  = rows * cols * sizeof(float);
    size_t out_bytes = cols * rows * sizeof(float);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, input.data(), in_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    matrix_transpose<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    std::vector<float> output(cols * rows);
    cudaMemcpy(output.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return output;
}

// 2×3 matrix:
// [ 1 2 3 ]      [ 1 4 ]
// [ 4 5 6 ]  →   [ 2 5 ]
//                [ 3 6 ]
TEST(MatrixTranspose, small_2x3) {
    std::vector<float> in = {1, 2, 3,
                             4, 5, 6};
    auto out = run_transpose(in, 2, 3);

    // out is 3×2 row-major: out[col * rows + row]
    EXPECT_FLOAT_EQ(out[0 * 2 + 0], 1.0f);  // (col=0, row=0)
    EXPECT_FLOAT_EQ(out[1 * 2 + 0], 2.0f);  // (col=1, row=0)
    EXPECT_FLOAT_EQ(out[2 * 2 + 0], 3.0f);  // (col=2, row=0)
    EXPECT_FLOAT_EQ(out[0 * 2 + 1], 4.0f);  // (col=0, row=1)
    EXPECT_FLOAT_EQ(out[1 * 2 + 1], 5.0f);  // (col=1, row=1)
    EXPECT_FLOAT_EQ(out[2 * 2 + 1], 6.0f);  // (col=2, row=1)
}

// Transpose 2 lần = identity
TEST(MatrixTranspose, double_transpose_is_identity) {
    int rows = 5, cols = 7;
    std::vector<float> in(rows * cols);
    for (int i = 0; i < rows * cols; i++) in[i] = (float)i;

    auto tmp = run_transpose(in, rows, cols);   // (rows×cols) → (cols×rows)
    auto out = run_transpose(tmp, cols, rows);  // (cols×rows) → (rows×cols)

    for (int i = 0; i < rows * cols; i++)
        EXPECT_FLOAT_EQ(out[i], in[i]);
}

// Square matrix: transpose(A)[i][j] == A[j][i]
TEST(MatrixTranspose, square_4x4) {
    int n = 4;
    std::vector<float> in(n * n);
    for (int i = 0; i < n * n; i++) in[i] = (float)i;

    auto out = run_transpose(in, n, n);

    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            EXPECT_FLOAT_EQ(out[c * n + r], in[r * n + c]);
}

// 1×N vector → N×1 column
TEST(MatrixTranspose, row_vector_to_column) {
    std::vector<float> in = {10.0f, 20.0f, 30.0f, 40.0f};
    auto out = run_transpose(in, 1, 4);
    // out is 4×1: out[col * 1 + 0] = col-th element
    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 20.0f);
    EXPECT_FLOAT_EQ(out[2], 30.0f);
    EXPECT_FLOAT_EQ(out[3], 40.0f);
}

// Larger matrix: 512×1024 — stress test 2D grid
TEST(MatrixTranspose, large_512x1024) {
    int rows = 512, cols = 1024;
    std::vector<float> in(rows * cols);
    for (int i = 0; i < rows * cols; i++) in[i] = (float)i;

    auto out = run_transpose(in, rows, cols);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            EXPECT_FLOAT_EQ(out[c * rows + r], in[r * cols + c]);
}
