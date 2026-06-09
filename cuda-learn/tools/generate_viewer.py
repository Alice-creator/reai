"""
Generate roadmap.html with embedded lesson viewer.
Run: python generate_viewer.py
"""
import json, re
from pathlib import Path

LESSONS_DIR = Path('/home/loc-dev/Projects/neon/cuda-learn/lessons')
OUT_FILE    = Path('/home/loc-dev/Projects/neon/cuda-learn/roadmap.html')

# ── Load scraped lessons ─────────────────────────────────────────────────────
lessons = {}
for f in LESSONS_DIR.glob('*.json'):
    if f.name == '_all.json': continue
    d = json.loads(f.read_text())
    slug = d.get('slug', f.stem)
    lessons[slug] = {
        'title':        d.get('title', slug),
        'section':      d.get('section', ''),
        'topic':        d.get('topic', ''),
        'description':  d.get('description', '')[:4000],
        'url':          d.get('url', ''),
        'html_content': d.get('html_content', ''),
    }

# ── CUDA kernel templates ────────────────────────────────────────────────────
TEMPLATES = {

'calculate-mean': '''\
// Mean = (1/n) * Σ x_i  →  two-step parallel reduction
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// Host: call reduce_sum twice, then mean = total / n
// int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
// reduce_sum<<<blocks, BLOCK_SIZE>>>(d_arr, d_partial, n);
// reduce_sum<<<1,     BLOCK_SIZE>>>(d_partial, d_sum, blocks);
// float mean = h_sum[0] / n;
''',

'calculate-variance-std': '''\
// Variance = (1/n) * Σ (x_i - mean)²  — two-pass approach
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

// Pass 1: compute mean (use reduce_sum from calculate-mean)

// Pass 2: compute variance
__global__ void reduce_variance(const float* arr, float* out, float mean, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = (i < n) ? (arr[i] - mean) : 0.0f;
    smem[tid] = diff * diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// std_dev = sqrtf(variance)
''',

'pearson-correlation': '''\
// Pearson r = Σ(xi-x̄)(yi-ȳ) / sqrt(Σ(xi-x̄)² * Σ(yi-ȳ)²)
// Need 3 reductions: sum_xy_centered, sum_x2, sum_y2
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void corr_products(const float* x, const float* y,
                               float mx, float my,
                               float* sum_xy, float* sum_x2, float* sum_y2,
                               int n) {
    __shared__ float sxy[BLOCK_SIZE], sx2[BLOCK_SIZE], sy2[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float dx = (i < n) ? x[i] - mx : 0.0f;
    float dy = (i < n) ? y[i] - my : 0.0f;
    sxy[tid] = dx * dy;
    sx2[tid] = dx * dx;
    sy2[tid] = dy * dy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sxy[tid] += sxy[tid + s];
            sx2[tid] += sx2[tid + s];
            sy2[tid] += sy2[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum_xy[blockIdx.x] = sxy[0];
        sum_x2[blockIdx.x] = sx2[0];
        sum_y2[blockIdx.x] = sy2[0];
    }
}
// r = total_xy / sqrtf(total_x2 * total_y2)
''',

'matrix-multiplication': '''\
// Naive GEMM: C[i][j] = Σ_k A[i][k] * B[k][j]
// Each thread computes one element of C
#include <cuda_runtime.h>

__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // j

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

// dim3 threads(16, 16);
// dim3 blocks((N+15)/16, (M+15)/16);
// matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
//
// TODO: optimize with shared memory tiling (Phase 8)
''',

'vector-norms': '''\
// L1 norm = Σ |x_i|
// L2 norm = sqrt(Σ x_i²)
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void norm_l1(const float* x, float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? fabsf(x[i]) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

__global__ void norm_l2_sq(const float* x, float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
// L2 = sqrtf(sum_of_squares)
''',

'gradient-computation': '''\
// Numerical gradient: ∂f/∂xi ≈ [f(x+h*ei) - f(x-h*ei)] / 2h
// Each thread computes gradient for one dimension
#include <cuda_runtime.h>

__device__ float f(const float* x, int n) {
    // TODO: replace with your target function
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return s;  // example: f(x) = sum(x_i²), grad = 2x
}

__global__ void numerical_gradient(const float* x, float* grad, int n, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Copy x to local (can't modify global for other threads)
    // For real use: allocate per-thread local copy
    float xi_orig = x[i];
    // central difference needs f(x+h) and f(x-h)
    // TODO: implement with device-side local copies
    grad[i] = 2.0f * xi_orig;  // placeholder for f(x)=sum(x²)
}
''',

'chain-rule-backprop': '''\
// Chain rule: dL/dx = dL/dy * dy/dx  — element-wise
// Used in backward pass of neural networks
#include <cuda_runtime.h>

// Backward through ReLU: grad_input = grad_output * (x > 0)
__global__ void relu_backward(const float* x, const float* grad_out,
                               float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        grad_in[i] = grad_out[i] * (x[i] > 0 ? 1.0f : 0.0f);
}

// Backward through sigmoid: grad = grad_out * σ(x) * (1 - σ(x))
__global__ void sigmoid_backward(const float* sigmoid_out, const float* grad_out,
                                  float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sigmoid_out[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}
''',

'hessian-computation': '''\
// Hessian diagonal: H_ii ≈ [f(x+h*ei) - 2f(x) + f(x-h*ei)] / h²
// Off-diagonal: H_ij ≈ [f(x+h*ei+h*ej) - f(x+h*ei) - f(x+h*ej) + f(x)] / h²
#include <cuda_runtime.h>

__global__ void hessian_diagonal(const float* x, float f0, float* diag,
                                   int n, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // diag[i] = (f(x+h*ei) - 2*f0 + f(x-h*ei)) / h²
    // TODO: evaluate f at perturbed points (requires device function for f)
    diag[i] = 0.0f;  // placeholder
}
''',

'taylor-approximation': '''\
// Taylor: f(x) ≈ Σ_k f^(k)(a)/k! * (x-a)^k
// Evaluate polynomial on GPU — each thread handles one x point
#include <cuda_runtime.h>

__global__ void taylor_eval(const float* coeffs, int degree,
                             const float* x_pts, float* y_pts,
                             float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dx  = x_pts[i] - a;
    float val = 0.0f;
    float pw  = 1.0f;  // (x-a)^k

    for (int k = 0; k <= degree; k++) {
        val += coeffs[k] * pw;
        pw  *= dx;
    }
    y_pts[i] = val;
}
''',

'manual-backprop': '''\
// Manual backprop for a 2-layer MLP: x → Linear → ReLU → Linear → Loss
#include <cuda_runtime.h>

// Forward: z = W*x + b, a = ReLU(z)
__global__ void linear_forward(const float* W, const float* x, const float* b,
                                float* z, int in, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron
    if (j >= out) return;
    float s = b[j];
    for (int i = 0; i < in; i++) s += W[j * in + i] * x[i];
    z[j] = s;
}

__global__ void relu_forward(const float* z, float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = fmaxf(0.0f, z[i]);
}

// Backward: dL/dW = dL/da * da/dz * dz/dW = delta * x^T
__global__ void grad_weights(const float* delta, const float* x,
                              float* dW, int in, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    for (int i = 0; i < in; i++)
        dW[j * in + i] = delta[j] * x[i];
}
''',

'convexity-check': '''\
// Check convexity: f is convex iff Hessian is positive semi-definite
// Simplified: check if all second derivatives >= 0 (for separable f)
#include <cuda_runtime.h>

__global__ void second_derivative(const float* x, float* d2f, int n, float h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // d²f/dx_i² ≈ (f(x+h*ei) - 2f(x) + f(x-h*ei)) / h²
    // Example for f(x) = x²: d²f/dx² = 2 > 0 → convex
    d2f[i] = 2.0f;  // TODO: replace with actual function evaluation
}

__global__ void is_positive(const float* arr, int* result, int n) {
    // result = 1 if all elements >= 0 (convex), else 0
    __shared__ int votes[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    votes[tid] = (i < n && arr[i] < 0.0f) ? 1 : 0;
    __syncthreads();
    // reduction: any negative → not convex
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) votes[tid] += votes[tid+s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, votes[0]);
}
''',

'sgd-minibatch': '''\
// SGD update: θ = θ - lr * grad
// Each thread updates one parameter
#include <cuda_runtime.h>

__global__ void sgd_update(float* params, const float* grads,
                            float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        params[i] -= lr * grads[i];
}

// With L2 regularization (weight decay):
__global__ void sgd_update_l2(float* params, const float* grads,
                               float lr, float wd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        params[i] -= lr * (grads[i] + wd * params[i]);
}

// Launch: sgd_update<<<(n+255)/256, 256>>>(d_params, d_grads, lr, n);
''',

'momentum-optimizer': '''\
// SGD + Momentum: v = β*v - lr*grad,  θ = θ + v
// Each thread handles one parameter
#include <cuda_runtime.h>

__global__ void momentum_update(float* params, float* velocity,
                                 const float* grads,
                                 float lr, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    velocity[i] = beta * velocity[i] - lr * grads[i];
    params[i]  += velocity[i];
}

// Nesterov variant: look-ahead gradient
__global__ void nesterov_update(float* params, float* velocity,
                                 const float* grads,
                                 float lr, float beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v_prev  = velocity[i];
    velocity[i]   = beta * v_prev - lr * grads[i];
    params[i]    += -beta * v_prev + (1 + beta) * velocity[i];
}
''',

'adam-implementation': '''\
// Adam: m = β1*m + (1-β1)*g
//       v = β2*v + (1-β2)*g²
//       m̂ = m/(1-β1^t),  v̂ = v/(1-β2^t)
//       θ = θ - lr * m̂/(sqrt(v̂) + ε)
#include <cuda_runtime.h>

__global__ void adam_update(float* params, float* m, float* v,
                             const float* grads,
                             float lr, float beta1, float beta2, float eps,
                             float beta1_t, float beta2_t,  // β1^t, β2^t
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g  = grads[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat = m[i] / (1.0f - beta1_t);
    float v_hat = v[i] / (1.0f - beta2_t);

    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}
// Typical: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
''',

'l1-l2-regularization': '''\
// L1 reg: loss += λ * Σ|θ_i|    grad += λ * sign(θ_i)
// L2 reg: loss += λ/2 * Σθ_i²  grad += λ * θ_i
#include <cuda_runtime.h>

__global__ void add_l2_grad(float* grads, const float* params,
                             float lambda, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grads[i] += lambda * params[i];
}

__global__ void add_l1_grad(float* grads, const float* params,
                             float lambda, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sign = (params[i] > 0) ? 1.0f : (params[i] < 0) ? -1.0f : 0.0f;
        grads[i] += lambda * sign;
    }
}

__global__ void l2_penalty(const float* params, float* loss_term,
                            float lambda, int n) {
    // Parallel reduction: sum params[i]²
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? params[i] * params[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) loss_term[blockIdx.x] = 0.5f * lambda * smem[0];
}
''',

'shannon-entropy': '''\
// Shannon Entropy: H(X) = -Σ p_i * log2(p_i)
// Parallel reduction: each thread computes p_i * log2(p_i)
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define LOG2E 1.4426950408f  // 1/ln(2)

__global__ void entropy_reduce(const float* probs, float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float p = (i < n && probs[i] > 0) ? probs[i] : 0.0f;
    smem[tid] = (p > 0) ? -p * logf(p) * LOG2E : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
// H(X) = sum of all partial results
''',

'cross-entropy-implementation': '''\
// Cross-Entropy: H(y, p) = -Σ y_i * log(p_i)
// y = true labels (one-hot), p = predicted probabilities
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define EPS 1e-7f  // avoid log(0)

__global__ void cross_entropy_reduce(const float* y, const float* p,
                                      float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float yi = (i < n) ? y[i] : 0.0f;
    float pi = (i < n) ? p[i] : 1.0f;
    smem[tid] = -yi * logf(pi + EPS);
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
''',

'kl-divergence': '''\
// KL Divergence: KL(P||Q) = Σ p_i * log(p_i / q_i)
// Note: KL is not symmetric — KL(P||Q) ≠ KL(Q||P)
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define EPS 1e-7f

__global__ void kl_divergence_reduce(const float* p, const float* q,
                                      float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float pi = (i < n && p[i] > 0) ? p[i] : 0.0f;
    float qi = (i < n) ? q[i] + EPS : EPS;
    smem[tid] = (pi > 0) ? pi * logf(pi / qi) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
''',

'mutual-information': '''\
// Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
// Or equivalently: I(X;Y) = Σ_xy p(x,y) * log[p(x,y)/(p(x)*p(y))]
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define EPS 1e-7f

__global__ void mi_joint(const float* pxy, const float* px, const float* py,
                          float* out, int nx, int ny) {
    // Each thread handles one (x,y) cell of the joint distribution
    int ix = blockIdx.y * blockDim.y + threadIdx.y;
    int iy = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= nx || iy >= ny) return;

    float p_joint = pxy[ix * ny + iy];
    float p_x     = px[ix];
    float p_y     = py[iy];

    // Contribution to MI: p(x,y) * log(p(x,y) / (p(x)*p(y)))
    float contrib = (p_joint > EPS) ?
        p_joint * logf(p_joint / (p_x * p_y + EPS)) : 0.0f;

    // TODO: reduce all contributions with atomicAdd
    atomicAdd(out, contrib);
}
''',

'information-gain': '''\
// Information Gain: IG(Y|X) = H(Y) - H(Y|X)
// H(Y|X) = Σ_x p(x) * H(Y|X=x) — used in decision trees
#include <cuda_runtime.h>
#define EPS 1e-7f

__device__ float entropy_of(const float* probs, int n) {
    float h = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] > EPS)
            h -= probs[i] * logf(probs[i]);
    }
    return h;
}

__global__ void conditional_entropy(const float* p_x, const float** p_y_given_x,
                                     float* H_Y_given_X, int nx, int ny) {
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (xi >= nx) return;
    H_Y_given_X[xi] = p_x[xi] * entropy_of(p_y_given_x[xi], ny);
}
// IG = H(Y) - sum(H_Y_given_X)
''',

'monte-carlo-pi': '''\
// Monte Carlo π estimation:
// Sample (x,y) ∈ [0,1]², count points inside unit circle (x²+y² < 1)
// π ≈ 4 * (points_inside / total_points)
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void mc_pi(unsigned long seed, int* inside_count, int n_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    int count = 0;
    for (int i = 0; i < n_per_thread; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x*x + y*y < 1.0f) count++;
    }
    atomicAdd(inside_count, count);
}
// pi ≈ 4.0f * inside_count / (n_threads * n_per_thread)
// Compile with: nvcc -lcurand monte_carlo.cu
''',

'bayes-theorem': '''\
// Bayes: P(H|E) = P(E|H) * P(H) / P(E)
// GPU use: update many hypotheses in parallel
#include <cuda_runtime.h>

__global__ void bayesian_update(const float* prior,          // P(H_i)
                                 const float* likelihood,     // P(E|H_i)
                                 float* posterior,            // P(H_i|E)
                                 float p_evidence,            // P(E)
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        posterior[i] = likelihood[i] * prior[i] / p_evidence;
}

// P(E) = Σ P(E|H_i) * P(H_i) — compute with parallel reduction first
__global__ void marginal_likelihood(const float* prior, const float* likelihood,
                                     float* out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? likelihood[i] * prior[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
''',

'conditional-probability': '''\
// P(A|B) = P(A ∩ B) / P(B)
// GPU: compute conditional probabilities for many events in parallel
#include <cuda_runtime.h>

// Each thread computes P(A_i | B) for one event A_i
__global__ void conditional_prob(const float* p_joint,  // P(A_i ∩ B)
                                  float p_b,             // P(B)
                                  float* p_cond,         // P(A_i | B)
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p_cond[i] = p_joint[i] / p_b;
}
''',

'expected-value-variance': '''\
// E[X] = Σ x_i * p_i  (weighted sum)
// Var(X) = E[X²] - (E[X])²
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void expected_value(const float* vals, const float* probs,
                                float* out, int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? vals[i] * probs[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
// Var(X) = E[X²] - E[X]² :
// Run expected_value with vals² to get E[X²], then subtract E[X]²
''',

'pmf-pdf-cdf': '''\
// PMF: P(X=k) — for discrete distributions
// PDF: f(x) — for continuous (probability density)
// CDF: F(x) = P(X ≤ x) — cumulative, parallel prefix sum!
#include <cuda_runtime.h>

// Evaluate Gaussian PDF in parallel: f(x) = exp(-(x-μ)²/2σ²) / (σ√2π)
__global__ void gaussian_pdf(const float* x, float* pdf, float mu, float sigma, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z   = (x[i] - mu) / sigma;
    float norm = 1.0f / (sigma * 2.5066f);  // 1/(σ√2π)
    pdf[i] = norm * expf(-0.5f * z * z);
}

// CDF via parallel prefix sum (scan)
__global__ void prefix_sum(const float* in, float* out, int n) {
    // TODO: implement parallel scan (Blelloch algorithm)
    // For now: sequential fallback
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = in[0];
        for (int i = 1; i < n; i++)
            out[i] = out[i-1] + in[i];
    }
}
''',

'index-2d-3d': '''\
// 2D Index Formula — matrix transpose
// Proof: in_index = row*cols+col, out_index = new_row*new_cols+new_col
//   new_row=col, new_cols=rows, new_col=row  →  out[col*rows+row]
#include <cuda_runtime.h>

__global__ void matrix_transpose(const float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int new_row  = col;   // row trong output = original col
    int new_col  = row;   // col trong output = original row
    int new_cols = rows;  // width của output = original rows

    if (row >= rows || col >= cols) return;
    out[new_row * new_cols + new_col] = in[row * cols + col];
}

// Launch:
// dim3 block(16, 16);
// dim3 grid((cols+15)/16, (rows+15)/16);
// matrix_transpose<<<grid, block>>>(d_in, d_out, rows, cols);

// 3D extension — batch processing:
// dim3 grid((cols+15)/16, (rows+15)/16, N);  // z = batch index
// int batch = blockIdx.z;
''',

'gram-schmidt': '''\
// Gram-Schmidt: orthogonalize a set of vectors
// Sequential in nature — each step depends on previous
// GPU parallelizes the dot products and projections
#include <cuda_runtime.h>

// Project v onto u: proj = (v·u)/(u·u) * u
__global__ void dot_product(const float* a, const float* b, float* out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    smem[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// Subtract projection: v = v - (v·u/u·u) * u
__global__ void subtract_projection(float* v, const float* u,
                                     float dot_vu, float dot_uu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] -= (dot_vu / dot_uu) * u[i];
}
''',

'eigenvalue-analysis': '''\
// Power iteration to find largest eigenvalue
// x_{k+1} = A * x_k / ||A * x_k||  →  converges to dominant eigenvector
#include <cuda_runtime.h>

// Matrix-vector product: y = A * x
__global__ void matvec(const float* A, const float* x, float* y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float s = 0;
    for (int j = 0; j < n; j++) s += A[row * n + j] * x[j];
    y[row] = s;
}

// Normalize vector in-place
__global__ void normalize(float* x, float norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] /= norm;
}

// Rayleigh quotient: λ ≈ (x^T A x) / (x^T x)  — use dot_product kernel
// Power iteration loop (host-side):
//   for k in range(max_iter):
//     matvec<<<>>>(A, x, y, n)       // y = A*x
//     norm = l2_norm(y)              // ||y||
//     normalize<<<>>>(y, norm, n)    // x = y/||y||
//   eigenvalue = rayleigh_quotient(A, x, n)
''',

'svd-decomposition': '''\
// SVD: A = U * Σ * V^T
// For GPU: use cuSOLVER (NVIDIA\'s official library)
#include <cuda_runtime.h>
#include <cusolverDn.h>

// cuSOLVER approach (recommended for production):
void svd_with_cusolver(float* d_A, int m, int n,
                        float* d_U, float* d_S, float* d_VT) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Query workspace size
    int lwork;
    cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);

    float *d_work, *d_rwork;
    cudaMalloc(&d_work,  lwork * sizeof(float));
    cudaMalloc(&d_rwork, (min(m,n)-1) * sizeof(float));

    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    // Compute SVD
    cusolverDnSgesvd(handle, \'A\', \'A\', m, n,
                     d_A, m, d_S, d_U, m, d_VT, n,
                     d_work, lwork, d_rwork, devInfo);

    // Compile: nvcc svd.cu -lcusolver
    cusolverDnDestroy(handle);
}
''',

'pca-from-scratch': '''\
// PCA steps:
// 1. Center data: X_c = X - mean(X)
// 2. Covariance: C = (1/n) * X_c^T * X_c
// 3. SVD of C: C = U * Σ * V^T
// 4. Principal components = columns of U
#include <cuda_runtime.h>

// Step 1: center each feature
__global__ void center_data(float* X, const float* means, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // sample
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // feature
    if (i < n && j < d)
        X[i * d + j] -= means[j];
}

// Step 2: covariance matrix C = X^T * X / n  (use matmul kernel)
// Then SVD via cuSOLVER (see svd-decomposition template)
// Step 4: project: Z = X_c * U[:, :k]  (top-k components)
__global__ void project(const float* X, const float* U, float* Z,
                         int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // sample
    int c = blockIdx.y * blockDim.y + threadIdx.y;  // component
    if (i >= n || c >= k) return;
    float s = 0;
    for (int j = 0; j < d; j++) s += X[i*d+j] * U[j*k+c];
    Z[i*k+c] = s;
}
''',

'numerical-limits': '''\
// Evaluate function near a limit point: lim_{x→a} f(x)
// Use many x values approaching a in parallel
#include <cuda_runtime.h>

__device__ float f(float x) {
    // TODO: replace with your function
    // Example: f(x) = sin(x)/x — limit as x→0 is 1
    return (fabsf(x) < 1e-7f) ? 1.0f : sinf(x) / x;
}

__global__ void eval_near_limit(float a, float* epsilons,
                                 float* f_values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a + epsilons[i];  // approach a from different distances
    f_values[i] = f(x);
}
// As epsilons → 0, f_values should converge to the limit
''',

'population-sample-stats': '''\
// Population: uses all N data points  →  divide by N
// Sample:     uses n < N points       →  divide by (n-1) [Bessel\'s correction]
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

// Bessel\'s correction for unbiased sample variance
__global__ void sample_variance(const float* arr, float* out, float mean,
                                 int n, bool is_sample) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = (i < n) ? arr[i] - mean : 0.0f;
    smem[tid]  = diff * diff;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) {
        // is_sample=true → divide by (n-1), false → divide by n
        out[blockIdx.x] = smem[0];  // host divides by n or n-1
    }
}
''',

'standard-error-calculation': '''\
// Standard Error: SE = σ / sqrt(n)
// σ = std dev of population (or sample std dev as estimate)
#include <cuda_runtime.h>

// After computing std_dev with reduce_variance kernel:
// SE = std_dev / sqrtf(n)
// This is a single scalar computation on the host side.

// For computing std dev of sample means (bootstrap):
__global__ void compute_sample_means(const float* data, float* means,
                                      int n_total, int sample_size,
                                      int n_samples) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_samples) return;

    // Each thread computes mean of one sample
    float sum = 0;
    int start = s * sample_size;
    for (int i = 0; i < sample_size && start + i < n_total; i++)
        sum += data[start + i];
    means[s] = sum / sample_size;
}
// SE = std_dev(means) — run variance reduction on means array
''',

'clt-simulation': '''\
// CLT: sum of n IID random vars → Normal distribution as n→∞
// Simulate: draw many samples, compute their means, show they\'re normal
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void simulate_sample_means(float* sample_means,
                                       int n_samples, int sample_size,
                                       unsigned long seed) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_samples) return;

    curandState state;
    curand_init(seed, s, 0, &state);

    // Draw sample_size uniform [0,1] values, compute mean
    float sum = 0;
    for (int i = 0; i < sample_size; i++)
        sum += curand_uniform(&state);
    sample_means[s] = sum / sample_size;
}
// As sample_size increases, sample_means becomes more normal
// Verify: compute mean ≈ 0.5, std ≈ 1/sqrt(12*sample_size)
''',

'ci-mean-known-sigma': '''\
// CI for mean (known σ): x̄ ± z_{α/2} * σ/√n
// z_{0.025} = 1.96 for 95% CI
#include <cuda_runtime.h>

// Bootstrap CI: compute many bootstrap sample means, take percentiles
__global__ void bootstrap_sample_mean(const float* data, float* boot_means,
                                       int n, int n_boot, unsigned long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_boot) return;

    curandState state;
    curand_init(seed, b, 0, &state);

    float sum = 0;
    for (int i = 0; i < n; i++) {
        int idx = (int)(curand_uniform(&state) * n);
        sum += data[idx % n];
    }
    boot_means[b] = sum / n;
}
// Sort boot_means, take 2.5th and 97.5th percentiles for 95% CI
''',

'hypothesis-setup': '''\
// Hypothesis testing: compute test statistic for many datasets in parallel
// z-score: z = (x̄ - μ₀) / (σ/√n)
#include <cuda_runtime.h>

// Compute z-score for each group/batch in parallel
__global__ void z_score_batch(const float* sample_means,  // x̄ per batch
                               const float* sample_stds,   // σ per batch
                               float* z_scores,
                               float mu0,                  // null hypothesis mean
                               const int* ns,              // sample sizes
                               int n_batches) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batches) return;
    float se = sample_stds[b] / sqrtf((float)ns[b]);
    z_scores[b] = (sample_means[b] - mu0) / se;
}
''',

'p-value-from-z': '''\
// P-value from z-score: p = 2 * (1 - Φ(|z|)) for two-tailed test
// Φ(z) = CDF of standard normal = 0.5 * (1 + erf(z/sqrt(2)))
#include <cuda_runtime.h>

__device__ float standard_normal_cdf(float z) {
    return 0.5f * (1.0f + erff(z / 1.41421356f));  // 1.414 = sqrt(2)
}

__global__ void p_values_from_z(const float* z_scores, float* p_values,
                                  int n, bool two_tailed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float phi = standard_normal_cdf(fabsf(z_scores[i]));
    p_values[i] = two_tailed ? 2.0f * (1.0f - phi) : (1.0f - phi);
}
''',

't-test-statistic': '''\
// One-sample t-test: t = (x̄ - μ₀) / (s/√n)
// Degrees of freedom: df = n - 1
#include <cuda_runtime.h>

// Compute t-statistic for many independent tests in parallel
__global__ void t_statistic_batch(const float* means,   // sample means
                                   const float* stds,    // sample std devs
                                   const int* ns,        // sample sizes
                                   float* t_stats,
                                   float mu0,            // null hypothesis
                                   int n_tests) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tests) return;
    float se    = stds[i] / sqrtf((float)ns[i]);
    t_stats[i]  = (means[i] - mu0) / se;
}
// Look up t-distribution CDF to get p-value (use SciPy on host)
''',

'ab-test-setup': '''\
// A/B test: compare conversion rates between two groups
// z = (p_A - p_B) / sqrt(p*(1-p)*(1/nA + 1/nB))  pooled proportion
#include <cuda_runtime.h>

// Simulate many A/B tests in parallel (power analysis)
__global__ void simulate_ab_test(float true_p_a, float true_p_b,
                                   int n_per_group, float* z_scores,
                                   int n_simulations, unsigned long seed) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_simulations) return;

    curandState state;
    curand_init(seed, s, 0, &state);

    int conv_a = 0, conv_b = 0;
    for (int i = 0; i < n_per_group; i++) {
        if (curand_uniform(&state) < true_p_a) conv_a++;
        if (curand_uniform(&state) < true_p_b) conv_b++;
    }
    float pa = (float)conv_a / n_per_group;
    float pb = (float)conv_b / n_per_group;
    float pp = (float)(conv_a + conv_b) / (2 * n_per_group);  // pooled
    float se = sqrtf(pp * (1-pp) * (2.0f / n_per_group));
    z_scores[s] = (pa - pb) / se;
}
''',

'mle-bernoulli': '''\
// MLE for Bernoulli: θ_MLE = (1/n) * Σ x_i = sample mean
// Log-likelihood: l(θ) = Σ[x_i*log(θ) + (1-x_i)*log(1-θ)]
#include <cuda_runtime.h>
#define EPS 1e-7f

// Evaluate log-likelihood for many θ values in parallel
__global__ void bernoulli_log_likelihood(const float* theta_vals,
                                          float n_success, float n_total,
                                          float* log_liks, int n_theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_theta) return;
    float t = theta_vals[i];
    t = fmaxf(EPS, fminf(1-EPS, t));  // clamp to (0,1)
    // l(θ) = k*log(θ) + (n-k)*log(1-θ)
    log_liks[i] = n_success * logf(t) + (n_total - n_success) * logf(1 - t);
}
// Max log_liks at θ = k/n = sample proportion
''',
}  # end TEMPLATES

# ── Canvas visualizations (JS bodies) ──────────────────────────────────────
VISUALIZATIONS = {

'calculate-mean': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const data=[2,3,4,4,5,5,5,6,6,7,8,9];
  const mean=data.reduce((a,b)=>a+b,0)/data.length;
  const counts=new Array(11).fill(0);
  data.forEach(v=>counts[v]++);
  const maxC=Math.max(...counts);
  const pad={l:40,r:20,t:20,b:40};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  const barW=cw/10;
  for(let v=0;v<=10;v++){
    const c=counts[v]; if(c===0) continue;
    const bh=(c/maxC)*ch*0.85;
    const x=pad.l+v*(cw/10), y=pad.t+ch-bh;
    ctx.fillStyle='#1e40af44'; ctx.fillRect(x+2,y,barW-4,bh);
    ctx.strokeStyle='#3b82f6'; ctx.lineWidth=1; ctx.strokeRect(x+2,y,barW-4,bh);
  }
  ctx.fillStyle=TXT; ctx.font='10px monospace'; ctx.textAlign='center';
  for(let v=0;v<=10;v+=2){ const x=pad.l+v*(cw/10)+barW/2; ctx.fillText(v,x,pad.t+ch+14); }
  ctx.textAlign='right';
  for(let c=0;c<=maxC;c++){ const y=pad.t+ch-(c/maxC)*ch*0.85; ctx.fillText(c,pad.l-4,y+3); }
  const mx=pad.l+mean*(cw/10)+barW/2;
  ctx.strokeStyle=BLU; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(mx,pad.t); ctx.lineTo(mx,pad.t+ch); ctx.stroke();
  ctx.fillStyle=BLU; ctx.textAlign='center'; ctx.font='bold 10px monospace';
  ctx.fillText('mu='+mean.toFixed(1),mx,pad.t+10);
  const sorted=[...data].sort((a,b)=>a-b);
  const med=(sorted[5]+sorted[6])/2;
  const medx=pad.l+med*(cw/10)+barW/2;
  ctx.strokeStyle=GRN; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(medx,pad.t); ctx.lineTo(medx,pad.t+ch); ctx.stroke();
  ctx.setLineDash([]);
  ctx.font='10px monospace'; ctx.textAlign='left';
  ctx.fillStyle=BLU; ctx.fillRect(W-100,8,10,3); ctx.fillText('Mean',W-87,13);
  ctx.fillStyle=GRN; ctx.fillRect(W-100,20,10,3); ctx.fillText('Median',W-87,25);
""",

'calculate-variance-std': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:24,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-4,xMax=4;
  function gauss(x,mu,sig){return Math.exp(-0.5*((x-mu)/sig)**2)/(sig*Math.sqrt(2*Math.PI));}
  function toCanvasX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toCanvasY(y,yMax){return pad.t+ch-(y/yMax)*ch*0.9;}
  const yMax=gauss(0,0,1)*1.1;
  ctx.fillStyle='rgba(96,165,250,0.08)';
  ctx.beginPath(); ctx.moveTo(toCanvasX(-2),pad.t+ch);
  for(let xi=-2;xi<=2;xi+=0.05){const y=gauss(xi,0,1);ctx.lineTo(toCanvasX(xi),toCanvasY(y,yMax));}
  ctx.lineTo(toCanvasX(2),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.fillStyle='rgba(96,165,250,0.18)';
  ctx.beginPath(); ctx.moveTo(toCanvasX(-1),pad.t+ch);
  for(let xi=-1;xi<=1;xi+=0.05){const y=gauss(xi,0,1);ctx.lineTo(toCanvasX(xi),toCanvasY(y,yMax));}
  ctx.lineTo(toCanvasX(1),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  for(let xi=xMin;xi<=xMax;xi+=0.05){
    const cx2=toCanvasX(xi),cy=toCanvasY(gauss(xi,0,1),yMax);
    xi===xMin?ctx.moveTo(cx2,cy):ctx.lineTo(cx2,cy);
  }
  ctx.stroke();
  const sigs=[-2,-1,0,1,2]; const labels=['-2s','-s','u','+s','+2s'];
  sigs.forEach((s,i)=>{
    ctx.strokeStyle=i===2?'#ffffff44':'#33415544'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(toCanvasX(s),pad.t); ctx.lineTo(toCanvasX(s),pad.t+ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
    ctx.fillText(labels[i],toCanvasX(s),pad.t+ch+14);
  });
  ctx.fillStyle='rgba(96,165,250,0.9)'; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('+-1s (68%)',toCanvasX(0),toCanvasY(gauss(0.5,0,1),yMax)-6);
  ctx.fillStyle='rgba(96,165,250,0.5)';
  ctx.fillText('+-2s (95%)',toCanvasX(1.6),toCanvasY(gauss(1.6,0,1),yMax)-18);
""",

'monte-carlo-pi': """\
  const BG='#0a0a0f',AXES='#334155',GRN='#4ade80',RED='#f87171',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad=30; const sz=Math.min(W,H)-pad*2;
  const ox=Math.floor((W-sz)/2), oy=Math.floor((H-sz)/2);
  ctx.strokeStyle=AXES; ctx.lineWidth=1; ctx.strokeRect(ox,oy,sz,sz);
  ctx.strokeStyle=BLU; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(ox,oy+sz,sz,-(Math.PI/2),0); ctx.stroke();
  let inside=0;
  for(let i=0;i<200;i++){
    const px=Math.abs(Math.sin(i*2.399)*0.5+Math.cos(i*1.618)*0.5);
    const py=Math.abs(Math.cos(i*2.399)*0.5+Math.sin(i*1.618)*0.5);
    const isIn=(px*px+py*py)<1;
    if(isIn) inside++;
    const cx=ox+px*sz, cy=oy+(1-py)*sz;
    ctx.fillStyle=isIn?GRN:RED;
    ctx.beginPath(); ctx.arc(cx,cy,2,0,Math.PI*2); ctx.fill();
  }
  const piEst=4*inside/200;
  ctx.fillStyle='#e2e8f0'; ctx.font='bold 13px monospace'; ctx.textAlign='center';
  ctx.fillText('pi ~ '+piEst.toFixed(3),W/2,pad-6);
  ctx.fillStyle=AXES; ctx.font='9px monospace';
  ctx.fillText('Green: inside   Red: outside',W/2,H-8);
""",

'pearson-correlation': """\
  const BG='#0a0a0f',TXT='#64748b',GRN='#4ade80',ORG='#fb923c';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const panels=[{r:0.95,label:'r~+1',col:GRN},{r:0,label:'r~0',col:TXT},{r:-0.95,label:'r~-1',col:ORG}];
  const pw=Math.floor(W/3)-10; const ph=H-40;
  panels.forEach((panel,pi)=>{
    const ox=pi*(W/3)+5; const oy=20;
    ctx.strokeStyle='#1e293b'; ctx.lineWidth=1; ctx.strokeRect(ox,oy,pw,ph);
    ctx.fillStyle=panel.col; ctx.font='bold 10px monospace'; ctx.textAlign='center';
    ctx.fillText(panel.label,ox+pw/2,oy-6);
    const n=15; const pts=[];
    for(let i=0;i<n;i++){
      const x=(i/(n-1))*2-1;
      const noise=(Math.sin(i*37.3+pi*100)*0.5)*(1-Math.abs(panel.r));
      const y=panel.r*x+noise;
      pts.push([x,y]);
    }
    function toX(v){return ox+4+(v+1.5)/(3)*(pw-8);}
    function toY(v){return oy+4+(1-(v+1.5)/(3))*(ph-8);}
    const mx=pts.reduce((a,p)=>a+p[0],0)/n;
    const my=pts.reduce((a,p)=>a+p[1],0)/n;
    const denom=pts.reduce((a,p)=>a+(p[0]-mx)**2,1e-9);
    const slope=pts.reduce((a,p)=>a+(p[0]-mx)*(p[1]-my),0)/denom;
    const intercept=my-slope*mx;
    ctx.strokeStyle=panel.col; ctx.lineWidth=1; ctx.setLineDash([3,2]);
    ctx.beginPath(); ctx.moveTo(toX(-1),toY(-1*slope+intercept)); ctx.lineTo(toX(1),toY(1*slope+intercept)); ctx.stroke();
    ctx.setLineDash([]);
    pts.forEach(p=>{ ctx.fillStyle=panel.col+'bb'; ctx.beginPath(); ctx.arc(toX(p[0]),toY(p[1]),3,0,Math.PI*2); ctx.fill(); });
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Pearson Correlation Coefficient',W/2,H-4);
""",

'shannon-entropy': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:24,b:40};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function Hfunc(p){if(p<=0||p>=1)return 0;return -p*Math.log2(p)-(1-p)*Math.log2(1-p);}
  function toX(p){return pad.l+p*cw;}
  function toY(h){return pad.t+ch-(h/1)*ch*0.85;}
  ctx.fillStyle='rgba(96,165,250,0.08)';
  ctx.beginPath(); ctx.moveTo(toX(0.01),pad.t+ch);
  for(let p=0.01;p<=0.99;p+=0.01){ctx.lineTo(toX(p),toY(Hfunc(p)));}
  ctx.lineTo(toX(0.99),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  let first=true;
  for(let p=0.01;p<=0.99;p+=0.005){
    const x=toX(p),y=toY(Hfunc(p));
    first?(ctx.moveTo(x,y),first=false):ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.strokeStyle='#fbbf2488'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(toX(0.5),pad.t); ctx.lineTo(toX(0.5),toY(1)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,toY(1)); ctx.lineTo(toX(0.5),toY(1)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#fbbf24'; ctx.font='bold 10px monospace'; ctx.textAlign='right';
  ctx.fillText('H=1 bit',pad.l-2,toY(1)+4);
  ctx.textAlign='center'; ctx.fillText('p=0.5',toX(0.5),pad.t+ch+13);
  ctx.fillStyle=TXT; ctx.font='9px monospace';
  [0,0.25,0.5,0.75,1].forEach(v=>ctx.fillText(v.toFixed(2),toX(v),pad.t+ch+27));
  ctx.textAlign='right';
  [0,0.5,1].forEach(v=>ctx.fillText(v.toFixed(1),pad.l-4,toY(v)+3));
  ctx.textAlign='center'; ctx.fillText('p (probability)',pad.l+cw/2,H-2);
""",

'kl-divergence': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',ORG='#fb923c',PNK='#f472b6';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:20,t:28,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-2,xMax=7;
  function gauss(x,mu,sig){return Math.exp(-0.5*((x-mu)/sig)**2)/(sig*Math.sqrt(2*Math.PI));}
  function toX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toY(y,yMax){return pad.t+ch-(y/yMax)*ch*0.9;}
  const yMax=0.55;
  ctx.fillStyle='rgba(244,114,182,0.15)';
  ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){
    const p=gauss(x,2,0.8),q=gauss(x,3,1.2);
    x===xMin?ctx.moveTo(toX(x),toY(Math.max(p,q),yMax)):ctx.lineTo(toX(x),toY(Math.max(p,q),yMax));
  }
  for(let x=xMax;x>=xMin;x-=0.05){
    ctx.lineTo(toX(x),toY(Math.min(gauss(x,2,0.8),gauss(x,3,1.2)),yMax));
  }
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){const cx=toX(x),cy=toY(gauss(x,2,0.8),yMax);x===xMin?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}
  ctx.stroke();
  ctx.strokeStyle=ORG; ctx.lineWidth=2; ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){const cx=toX(x),cy=toY(gauss(x,3,1.2),yMax);x===xMin?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}
  ctx.stroke();
  ctx.font='10px monospace'; ctx.textAlign='left';
  ctx.fillStyle=BLU; ctx.fillText('P (true, mu=2, s=0.8)',pad.l+4,pad.t+12);
  ctx.fillStyle=ORG; ctx.fillText('Q (approx, mu=3, s=1.2)',pad.l+4,pad.t+26);
  ctx.fillStyle=PNK; ctx.fillText('KL divergence area',pad.l+4,pad.t+40);
""",

'matrix-multiplication': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',GRN='#4ade80',YEL='#fbbf24';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const n=3; const cellSz=28; const gap=18;
  const mw=n*cellSz;
  const ox=Math.floor((W-(mw*3+gap*2+32))/2); const oy=Math.floor((H-mw)/2)-10;
  function drawMatrix(ox2,oy2,hlRow,hlCol,label,borderCol){
    for(let r=0;r<n;r++) for(let c=0;c<n;c++){
      const x=ox2+c*cellSz, y=oy2+r*cellSz;
      let bg='#0f172a';
      if(hlRow!==null&&r===hlRow) bg='rgba(96,165,250,0.2)';
      if(hlCol!==null&&c===hlCol) bg='rgba(74,222,128,0.2)';
      if(hlRow!==null&&hlCol!==null&&r===hlRow&&c===hlCol) bg='rgba(251,191,36,0.35)';
      ctx.fillStyle=bg; ctx.fillRect(x+1,y+1,cellSz-2,cellSz-2);
      ctx.strokeStyle=AXES; ctx.lineWidth=1; ctx.strokeRect(x,y,cellSz,cellSz);
      ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
      ctx.fillText(r*n+c+1,x+cellSz/2,y+cellSz/2+4);
    }
    if(borderCol){ctx.strokeStyle=borderCol;ctx.lineWidth=2;ctx.strokeRect(ox2,oy2,mw,mw);}
    ctx.fillStyle=TXT; ctx.font='bold 11px monospace'; ctx.textAlign='center';
    ctx.fillText(label,ox2+mw/2,oy2-6);
  }
  const ax=ox, bx=ox+mw+gap+16, cx2=bx+mw+gap+16;
  drawMatrix(ax,oy,0,null,'A',BLU);
  drawMatrix(bx,oy,null,0,'B',GRN);
  drawMatrix(cx2,oy,0,0,'C',YEL);
  ctx.fillStyle='#94a3b8'; ctx.font='bold 16px sans-serif'; ctx.textAlign='center';
  ctx.fillText('x',ox+mw+gap/2+8,oy+mw/2+6);
  ctx.fillText('=',bx+mw+gap/2+8,oy+mw/2+6);
  ctx.fillStyle=YEL; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('C[i][j] = sum_k A[i][k]*B[k][j]',W/2,oy+mw+22);
""",

'adam-implementation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',RED='#f87171',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:28,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const N=100;
  function sgdLoss(i){return 0.9*Math.exp(-i/120)+0.1+Math.sin(i*0.8)*0.06*(1-i/150);}
  function momLoss(i){return 0.85*Math.exp(-i/80)+0.08+Math.sin(i*0.4)*0.025*(1-i/120);}
  function adamLoss(i){return 0.8*Math.exp(-i/35)+0.05+Math.sin(i*0.2)*0.01;}
  function toX(i){return pad.l+(i/N)*cw;}
  function toY(v){return pad.t+ch-Math.min(1,Math.max(0,(v/1.05)*ch*0.88));}
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  [[sgdLoss,RED,'SGD'],[momLoss,ORG,'Momentum'],[adamLoss,GRN,'Adam']].forEach(([fn,col,lbl],li)=>{
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
    for(let i=0;i<=N;i++){const x=toX(i),y=toY(fn(i));i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.stroke();
    ctx.fillStyle=col; ctx.font='10px monospace'; ctx.textAlign='right';
    ctx.fillText(lbl,pad.l+cw-4,pad.t+16+li*14);
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Iterations (0-100)',pad.l+cw/2,H-4);
  [0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v.toFixed(1),pad.l-4,toY(v)+3);});
  ctx.save(); ctx.translate(12,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.fillText('Loss',0,0); ctx.restore();
""",

'taylor-approximation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',WHT='#e2e8f0',RED='#f87171',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:20,t:24,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-Math.PI,xMax=Math.PI,yMin=-1.5,yMax=1.5;
  function toX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toY(y){return pad.t+ch-(y-yMin)/(yMax-yMin)*ch;}
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,toY(0)); ctx.lineTo(pad.l+cw,toY(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX(0),pad.t); ctx.lineTo(toX(0),pad.t+ch); ctx.stroke();
  function drawFn(fn,col,dash){
    ctx.strokeStyle=col; ctx.lineWidth=1.5;
    if(dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
    ctx.beginPath(); let first=true;
    for(let x=xMin;x<=xMax;x+=0.03){
      const y=fn(x); if(isNaN(y)||!isFinite(y)) continue;
      const cy=toY(Math.max(yMin,Math.min(yMax,y)));
      first?(ctx.moveTo(toX(x),cy),first=false):ctx.lineTo(toX(x),cy);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }
  drawFn(x=>Math.sin(x),WHT);
  drawFn(x=>x,RED,[4,3]);
  drawFn(x=>x-x**3/6,ORG,[4,3]);
  drawFn(x=>x-x**3/6+x**5/120,GRN,[4,3]);
  ctx.font='9px monospace'; ctx.textAlign='left';
  [[WHT,'sin(x)'],[RED,'deg 1: x'],[ORG,'deg 3'],[GRN,'deg 5']].forEach(([c,l],i)=>{
    ctx.fillStyle=c; ctx.fillText(l,pad.l+4,pad.t+12+i*12);
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Taylor Approximations of sin(x)',W/2,H-4);
""",

'convexity-check': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',GRN='#4ade80',RED='#f87171';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const half=Math.floor(W/2)-4;
  const pad={l:20,r:10,t:28,b:36};
  function drawPanel(ox,ow,fn,col,label){
    const ch=H-pad.t-pad.b;
    const xMin=-2,xMax=2;
    const vals=[]; for(let x=xMin;x<=xMax;x+=0.1) vals.push(fn(x));
    const yMin2=Math.min(...vals)-0.2, yMax2=Math.max(...vals)+0.2;
    function toX(x){return ox+pad.l+(x-xMin)/(xMax-xMin)*(ow-pad.l-pad.r);}
    function toY(y){return pad.t+ch-(y-yMin2)/(yMax2-yMin2)*ch;}
    ctx.strokeStyle=AXES; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(ox+pad.l,pad.t); ctx.lineTo(ox+pad.l,pad.t+ch); ctx.lineTo(ox+ow-pad.r,pad.t+ch); ctx.stroke();
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath(); let first=true;
    for(let x=xMin;x<=xMax;x+=0.04){
      const y=fn(x); const cx=toX(x),cy=toY(y);
      first?(ctx.moveTo(cx,cy),first=false):ctx.lineTo(cx,cy);
    }
    ctx.stroke();
    const x1=-1.5,x2=1.5;
    ctx.strokeStyle=col+'88'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(toX(x1),toY(fn(x1))); ctx.lineTo(toX(x2),toY(fn(x2))); ctx.stroke();
    ctx.setLineDash([]);
    [x1,x2].forEach(x=>{ctx.fillStyle=col;ctx.beginPath();ctx.arc(toX(x),toY(fn(x)),4,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle=col; ctx.font='bold 10px monospace'; ctx.textAlign='center';
    ctx.fillText(label,ox+ow/2,pad.t-8);
  }
  drawPanel(0,half,x=>x*x+0.1,GRN,'Convex (check)');
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(half+4,0); ctx.lineTo(half+4,H); ctx.stroke();
  drawPanel(half+8,half,x=>Math.sin(x*2)*0.4+x*x*0.1,RED,'Non-Convex (x)');
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('chord above = convex',W/4,H-4);
  ctx.fillText('chord below = non-convex',W*3/4,H-4);
""",

'gradient-computation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',RED='#f87171',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const cx=Math.floor(W/2), cy=Math.floor(H/2);
  const colors=['#1e3a5f','#1e4080','#1e4d99','#1a5db3','#1a6bcc'];
  for(let i=0;i<5;i++){
    const rx=(i+1)*28, ry=(i+1)*18;
    ctx.strokeStyle=colors[i]; ctx.lineWidth=1;
    ctx.beginPath(); ctx.ellipse(cx,cy,rx,ry,0,0,Math.PI*2); ctx.stroke();
  }
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(30,cy); ctx.lineTo(W-20,cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,20); ctx.lineTo(cx,H-20); ctx.stroke();
  const scaleX=36, scaleY=24;
  const px=cx+1.5*scaleX, py=cy-1.0*scaleY;
  ctx.fillStyle=RED; ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
  const gx=3,gy=2; const glen=Math.sqrt(gx*gx+gy*gy);
  const arrowLen=45;
  const dx=(gx/glen)*arrowLen, dy=-(gy/glen)*arrowLen;
  ctx.strokeStyle=GRN; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+dx,py+dy); ctx.stroke();
  const angle=Math.atan2(dy,dx);
  ctx.fillStyle=GRN; ctx.beginPath();
  ctx.moveTo(px+dx,py+dy);
  ctx.lineTo(px+dx-10*Math.cos(angle-0.4),py+dy-10*Math.sin(angle-0.4));
  ctx.lineTo(px+dx-10*Math.cos(angle+0.4),py+dy-10*Math.sin(angle+0.4));
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#f472b6'; ctx.lineWidth=1.5; ctx.setLineDash([3,2]);
  ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px-dx*0.7,py-dy*0.7); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle=GRN; ctx.font='9px monospace'; ctx.textAlign='left';
  ctx.fillText('grad-f (uphill)',px+dx+4,py+dy);
  ctx.fillStyle='#f472b6'; ctx.textAlign='right';
  ctx.fillText('-grad-f (descent)',px-dx*0.7-4,py-dy*0.7);
  ctx.fillStyle=TXT; ctx.textAlign='center';
  ctx.fillText('f(x,y) = x^2 + y^2  contours',W/2,H-8);
""",

'sgd-minibatch': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:24,t:28,b:48};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function lossFn(x){return (x-0.5)**2*0.8+0.05+Math.sin(x*18)*0.015+Math.cos(x*11)*0.012;}
  function toX(x){return pad.l+x*cw;}
  function toY(v,yMin,yMax){return pad.t+ch-(v-yMin)/(yMax-yMin)*ch;}
  const steps=15; let wx=0.1;
  const lr=0.12; const pts=[{x:wx,y:lossFn(wx)}];
  for(let i=0;i<steps-1;i++){
    const grad=(lossFn(wx+0.001)-lossFn(wx-0.001))/0.002+Math.sin(i*3.7)*0.04;
    wx=Math.max(0.02,Math.min(0.98,wx-lr*grad));
    pts.push({x:wx,y:lossFn(wx)});
  }
  const yMin=0, yMax=Math.max(...pts.map(p=>p.y))*1.3;
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle='#334155'; ctx.lineWidth=2; ctx.beginPath();
  for(let x=0;x<=1;x+=0.005){
    const cx=toX(x),cy=toY(lossFn(x),yMin,yMax);
    x===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
  }
  ctx.stroke();
  for(let i=0;i<pts.length-1;i++){
    const p1=pts[i],p2=pts[i+1];
    ctx.strokeStyle=ORG+'88'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(toX(p1.x),toY(p1.y,yMin,yMax)); ctx.lineTo(toX(p2.x),toY(p2.y,yMin,yMax)); ctx.stroke();
    ctx.fillStyle=ORG; ctx.beginPath(); ctx.arc(toX(p1.x),toY(p1.y,yMin,yMax),3,0,Math.PI*2); ctx.fill();
  }
  const last=pts[pts.length-1];
  ctx.fillStyle=GRN; ctx.beginPath(); ctx.arc(toX(last.x),toY(last.y,yMin,yMax),5,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fbbf24'; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('w = w - lr * grad_L',W/2,H-6);
  ctx.fillStyle=TXT; ctx.font='9px monospace';
  ctx.fillText('SGD steps: orange=path, green=minimum',W/2,H-18);
""",

}  # end VISUALIZATIONS

# -- Build VISUALIZATIONS JS object ------------------------------------------
def build_viz_js():
    lines = ['const VISUALIZATIONS = {']
    for slug, body in VISUALIZATIONS.items():
        lines.append(f"  '{slug}': (canvas,W,H,ctx)=>{{")  
        lines.append(body.rstrip('\n'))
        lines.append('  },')
    lines.append('};')
    return '\n'.join(lines)

# Generic fallback template
GENERIC_TEMPLATE = '''\
// {title}
// Section: {section} | Topic: {topic}
//
// GPU Implementation Plan:
//   1. Identify the core computation pattern:
//      - Is it a reduction? (sum, max, mean)
//      - Is it element-wise? (activation functions)
//      - Is it a matrix op? (matmul, transpose)
//   2. Choose appropriate thread/block layout
//   3. Implement kernel with shared memory if needed
//   4. Verify correctness on small example first
//
#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void kernel_{slug}(/* TODO: add parameters */) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: implement {title} on GPU
}}

// Launch:
// int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
// kernel_{slug}<<<blocks, BLOCK_SIZE>>>(/* args */);
// cudaDeviceSynchronize();
'''

def get_template(slug, lesson):
    if slug in TEMPLATES:
        return TEMPLATES[slug]
    return GENERIC_TEMPLATE.format(
        title=lesson.get('title', slug),
        section=lesson.get('section', ''),
        topic=lesson.get('topic', ''),
        slug=slug.replace('-', '_'),
    )

# ── Build JS data objects ────────────────────────────────────────────────────
lessons_js_entries = []
for slug, data in lessons.items():
    tmpl = get_template(slug, data)
    entry = {
        'slug':         slug,
        'title':        data['title'],
        'section':      data['section'],
        'topic':        data['topic'],
        'description':  data['description'],
        'url':          data['url'],
        'template':     tmpl,
        'html_content': data.get('html_content', ''),
    }
    lessons_js_entries.append(entry)

lessons_js = 'const LESSONS = ' + json.dumps(
    {e['slug']: e for e in lessons_js_entries},
    ensure_ascii=False, indent=None
) + ';'

# ── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CUDA × ML Math Roadmap</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
  <style>
    :root {
      --bg: #0f172a; --surface: #1e293b; --surf2: #263347;
      --text: #e2e8f0; --muted: #64748b; --border: #334155; --done: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    header { position: sticky; top: 0; z-index: 50; background: rgba(15,23,42,.96); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 14px 28px; flex-shrink: 0; }
    .hrow { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .htitle { font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .hsub { font-size: .7rem; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px; font-weight: 400; }
    .hprog { font-size: .82rem; color: var(--muted); font-variant-numeric: tabular-nums; }
    .ptrack { height: 4px; background: var(--border); border-radius: 999px; overflow: hidden; }
    .pfill { height: 100%; width: 0%; background: linear-gradient(90deg,#10b981,#06b6d4,#8b5cf6); border-radius: 999px; transition: width .5s; }

    .app-body { display: flex; flex: 1; overflow: hidden; }
    #roadmap-scroll { flex: 1; overflow-y: auto; transition: flex 0.3s ease; }
    .app-body.panel-open #roadmap-scroll { flex: 0 0 54%; }

    #wrapper { max-width: 960px; margin: 0 auto; padding: 48px 28px 100px; position: relative; }
    #lines { position: absolute; top: 0; left: 0; width: 100%; pointer-events: none; overflow: visible; }

    .section { margin-bottom: 72px; }
    .phase-node { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 20px 36px; border-radius: 16px; border: 2px solid; margin: 0 auto; max-width: 440px; position: relative; z-index: 2; }
    .pnum { font-size: .65rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 4px; opacity: .7; }
    .pname { font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }
    .psub { font-size: .72rem; opacity: .65; }
    .pcount { font-size: .72rem; margin-top: 6px; opacity: .5; }

    .topic-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 40px; padding: 0 12px; }

    .topic { display: flex; align-items: center; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 9px 14px; font-size: .81rem; font-weight: 500; user-select: none; position: relative; z-index: 2; transition: background .15s, transform .15s, box-shadow .15s, border-color .2s; }
    .topic.has-lesson { cursor: pointer; }
    .topic.has-lesson:hover { background: var(--surf2); transform: translateY(-2px); box-shadow: 0 5px 18px rgba(0,0,0,.4); }
    .topic.done { border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.07); color: var(--muted); text-decoration: line-through; text-decoration-color: rgba(16,185,129,.5); }
    .topic.selected { outline: 2px solid #60a5fa; outline-offset: 2px; }
    .topic.pop { animation: pop .25s ease; }
    @keyframes pop { 0%{transform:scale(1)} 45%{transform:scale(1.08)} 100%{transform:scale(1)} }
    .dot { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 9px; color: transparent; flex-shrink: 0; transition: all .2s; cursor: pointer; }
    .dot:hover { border-color: var(--done); }
    .topic.done .dot { background: var(--done); border-color: var(--done); color: #fff; }

    .reset-btn { display: block; margin: 24px auto 0; padding: 8px 22px; background: transparent; border: 1px solid var(--border); border-radius: 6px; color: var(--muted); font-size: .78rem; cursor: pointer; transition: all .2s; }
    .reset-btn:hover { border-color: #ef4444; color: #ef4444; }

    #lesson-pane { width: 0; overflow: hidden; border-left: 1px solid var(--border); background: #0d1117; display: flex; flex-direction: column; transition: width 0.3s ease; flex-shrink: 0; }
    .app-body.panel-open #lesson-pane { width: 46%; }
    .lesson-header { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .lesson-header-info { flex: 1; min-width: 0; }
    .lesson-header h2 { font-size: .9rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .lesson-section-badge { font-size: .65rem; padding: 2px 8px; border-radius: 4px; background: var(--surface); color: var(--muted); margin-top: 3px; display: inline-block; }
    .close-btn { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 1.1rem; padding: 4px 8px; border-radius: 4px; flex-shrink: 0; }
    .close-btn:hover { background: var(--surface); color: var(--text); }

    .tabs { display: flex; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .tab { padding: 10px 18px; font-size: .8rem; font-weight: 600; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; transition: color .15s; }
    .tab:hover { color: #94a3b8; }
    .tab.active { color: #60a5fa; border-bottom-color: #60a5fa; }
    .tab-link { margin-left: auto; padding: 10px 14px; font-size: .75rem; color: var(--border); text-decoration: none; }
    .tab-link:hover { color: #60a5fa; }
    .tab-content { flex: 1; overflow-y: auto; display: none; }
    .tab-content.active { display: block; }

    .theory-body { padding: 18px; font-size: .82rem; line-height: 1.75; color: #94a3b8; }
    .theory-body p { margin-bottom: 10px; }
    .def-box { background: var(--bg); border-left: 3px solid #60a5fa; padding: 10px 14px; border-radius: 4px; margin: 10px 0; font-size: .8rem; }
    .ex-box  { background: var(--bg); border-left: 3px solid #4ade80; padding: 10px 14px; border-radius: 4px; margin: 10px 0; font-size: .8rem; }
    .section-hdr { color: var(--text); font-weight: 700; font-size: .85rem; margin: 14px 0 6px; }
    .theory-body strong { color: #cbd5e1; }
    .theory-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
    .theory-topic { font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--border); margin-bottom: 14px; }

    .code-pane { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
    .code-toolbar { display: flex; align-items: center; gap: 8px; }
    .code-filename { font-size: .72rem; color: var(--muted); font-family: monospace; background: var(--bg); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); }
    .copy-btn { margin-left: auto; background: var(--surface); border: 1px solid var(--border); color: #94a3b8; font-size: .72rem; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
    .copy-btn:hover { background: var(--border); color: var(--text); }
    .copy-btn.copied { background: #14532d33; color: #4ade80; border-color: #166534; }
    pre { background: #060609; border: 1px solid var(--border); border-radius: 8px; padding: 16px; overflow-x: auto; font-size: .75rem; line-height: 1.6; font-family: 'Cascadia Code','Fira Code',monospace; color: #94a3b8; white-space: pre; }
    .kw{color:#c792ea;} .ty{color:#82aaff;} .fn{color:#82aaff;} .cm{color:#546e7a;font-style:italic;} .nu{color:#f78c6c;} .pp{color:#c792ea;} .deco{color:#ffcb6b;}
    .lesson-footer { padding: 12px 18px; border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; flex-shrink: 0; }
    .done-btn { flex: 1; padding: 8px; border-radius: 8px; font-size: .8rem; font-weight: 600; cursor: pointer; border: 1px solid #166534; background: #14532d33; color: #4ade80; transition: background .15s; }
    .done-btn:hover { background: #14532d66; }
    .done-btn.already-done { background: #14532d; color: #4ade80; cursor: default; }
    #viz-canvas { width: 100%; max-height: 220px; display: none; border-bottom: 1px solid var(--border); background: #0a0a0f; }
  </style>
</head>
<body>

<header>
  <div class="hrow">
    <div class="htitle">⚡ CUDA × ML Math <span class="hsub">RTX 3060 · CUDA 12.8</span></div>
    <span class="hprog" id="prog-text">0 / 0 completed</span>
  </div>
  <div class="ptrack"><div class="pfill" id="prog-fill"></div></div>
</header>

<div class="app-body" id="app-body">
  <div id="roadmap-scroll">
    <div id="wrapper">
      <svg id="lines"></svg>
      <div id="graph"></div>
    </div>
  </div>

  <div id="lesson-pane">
    <div class="lesson-header">
      <div class="lesson-header-info">
        <h2 id="lp-title">—</h2>
        <span class="lesson-section-badge" id="lp-section"></span>
      </div>
      <button class="close-btn" onclick="closePanel()">✕</button>
    </div>
    <div class="tabs">
      <div class="tab active" onclick="switchTab('theory')">📖 Theory</div>
      <div class="tab" onclick="switchTab('code')">⚙️ CUDA Kernel</div>
      <a id="lp-link" href="#" target="_blank" class="tab-link">↗ tensortonic</a>
    </div>
    <div id="tab-theory" class="tab-content active">
      <div id="lp-viz-html" style="display:none;border-bottom:1px solid var(--border);"></div>
      <canvas id="viz-canvas" width="480" height="220"></canvas>
      <div class="theory-body">
        <div class="theory-topic" id="lp-topic"></div>
        <div class="theory-title" id="lp-theory-title"></div>
        <div id="lp-description"></div>
      </div>
    </div>
    <div id="tab-code" class="tab-content">
      <div class="code-pane">
        <div class="code-toolbar">
          <span class="code-filename" id="lp-filename">kernel.cu</span>
          <button class="copy-btn" id="copy-btn" onclick="copyCode()">Copy</button>
        </div>
        <pre id="lp-code"></pre>
      </div>
    </div>
    <div class="lesson-footer">
      <button class="done-btn" id="done-btn" onclick="markDone()">✓ Mark as Done</button>
    </div>
  </div>
</div>

<script>
LESSONS_PLACEHOLDER

const PHASES = [
  {n:1, title:'GPU Foundations', color:'#10b981', sub:'Thread model · first kernels · memory model', topics:[
    {name:'CUDA Kernel Foundations', problems:[
      {t:'Vector Addition',    s:'done', slug:'vector-addition'},
      {t:'Vector Subtraction', s:'done', slug:'vector-subtraction'},
      {t:'ReLU',               s:'done', slug:'relu'},
      {t:'Sigmoid',            s:'done', slug:'sigmoid'},
      {t:'Tanh',               s:'done', slug:'tanh'},
      {t:'Leaky ReLU',         s:'done', slug:'leaky-relu'},
      {t:'GELU',               s:'done', slug:'gelu'},
      {t:'Swish',              s:'done', slug:'swish'},
    ]},
    {name:'CUDA Concepts', problems:[
      {t:'Thread / Block / Grid', s:'done',   slug:null},
      {t:'2D / 3D Index Formula', s:'done',   slug:'index-2d-3d'},
      {t:'Memory Coalescing',     s:'locked', slug:null},
      {t:'Shared Memory',         s:'locked', slug:null},
    ]},
  ]},
  {n:2, title:'Statistics on GPU', color:'#3b82f6', sub:'Parallel reduction → statistical kernels', topics:[
    {name:'Descriptive Statistics', problems:[
      {t:'Calculate Mean',        s:'locked', slug:'calculate-mean'},
      {t:'Calculate Variance',    s:'locked', slug:'calculate-variance-std'},
      {t:'Population vs Sample',  s:'locked', slug:'population-sample-stats'},
    ]},
    {name:'Sampling & Inference', problems:[
      {t:'Standard Error',        s:'locked', slug:'standard-error-calculation'},
      {t:'Central Limit Theorem', s:'locked', slug:'clt-simulation'},
      {t:'Confidence Intervals',  s:'locked', slug:'ci-mean-known-sigma'},
    ]},
    {name:'Hypothesis Testing', problems:[
      {t:'Hypothesis Setup',      s:'locked', slug:'hypothesis-setup'},
      {t:'P-Value from Z',        s:'locked', slug:'p-value-from-z'},
      {t:'T-Test Statistic',      s:'locked', slug:'t-test-statistic'},
      {t:'A/B Test Setup',        s:'locked', slug:'ab-test-setup'},
    ]},
    {name:'Correlation & MLE', problems:[
      {t:'Pearson Correlation',   s:'locked', slug:'pearson-correlation'},
      {t:'MLE Bernoulli',         s:'locked', slug:'mle-bernoulli'},
    ]},
  ]},
  {n:3, title:'Linear Algebra on GPU', color:'#8b5cf6', sub:'Matrix kernels · decompositions · geometric ops', topics:[
    {name:'Matrix Operations', problems:[
      {t:'Matrix Multiplication', s:'locked', slug:'matrix-multiplication'},
      {t:'Vector Norms',          s:'locked', slug:'vector-norms'},
      {t:'Gram-Schmidt',          s:'locked', slug:'gram-schmidt'},
    ]},
    {name:'Decompositions', problems:[
      {t:'Eigenvalue Analysis',   s:'locked', slug:'eigenvalue-analysis'},
      {t:'SVD Decomposition',     s:'locked', slug:'svd-decomposition'},
      {t:'PCA from Scratch',      s:'locked', slug:'pca-from-scratch'},
    ]},
  ]},
  {n:4, title:'Probability on GPU', color:'#14b8a6', sub:'Sampling · Monte Carlo · Bayesian kernels', topics:[
    {name:'Probability Kernels', problems:[
      {t:'Conditional Probability',   s:'locked', slug:'conditional-probability'},
      {t:'PMF / PDF / CDF',           s:'locked', slug:'pmf-pdf-cdf'},
      {t:'Expected Value & Variance', s:'locked', slug:'expected-value-variance'},
      {t:'Bayes Theorem',             s:'locked', slug:'bayes-theorem'},
      {t:'Monte Carlo Pi',            s:'locked', slug:'monte-carlo-pi'},
    ]},
  ]},
  {n:5, title:'Calculus & Autograd', color:'#f97316', sub:'Gradients · backprop · numerical differentiation', topics:[
    {name:'Calculus Kernels', problems:[
      {t:'Numerical Limits',     s:'locked', slug:'numerical-limits'},
      {t:'Gradient Computation', s:'locked', slug:'gradient-computation'},
      {t:'Chain Rule Backprop',  s:'locked', slug:'chain-rule-backprop'},
      {t:'Hessian Computation',  s:'locked', slug:'hessian-computation'},
      {t:'Taylor Approximation', s:'locked', slug:'taylor-approximation'},
      {t:'Manual Backprop',      s:'locked', slug:'manual-backprop'},
    ]},
  ]},
  {n:6, title:'Optimization on GPU', color:'#ec4899', sub:'SGD · Adam · regularization kernels', topics:[
    {name:'Optimizer Kernels', problems:[
      {t:'Convexity Check',      s:'locked', slug:'convexity-check'},
      {t:'SGD Mini-Batch',       s:'locked', slug:'sgd-minibatch'},
      {t:'Momentum Optimizer',   s:'locked', slug:'momentum-optimizer'},
      {t:'Adam Optimizer',       s:'locked', slug:'adam-implementation'},
      {t:'L1/L2 Regularization', s:'locked', slug:'l1-l2-regularization'},
    ]},
  ]},
  {n:7, title:'Information Theory on GPU', color:'#eab308', sub:'Entropy · KL divergence · mutual information', topics:[
    {name:'Information Theory Kernels', problems:[
      {t:'Shannon Entropy',   s:'locked', slug:'shannon-entropy'},
      {t:'Cross-Entropy',     s:'locked', slug:'cross-entropy-implementation'},
      {t:'KL Divergence',     s:'locked', slug:'kl-divergence'},
      {t:'Mutual Information',s:'locked', slug:'mutual-information'},
      {t:'Information Gain',  s:'locked', slug:'information-gain'},
    ]},
  ]},
  {n:8, title:'Performance & Profiling', color:'#06b6d4', sub:'Tiling · shared memory · ncu · occupancy', topics:[
    {name:'Performance', problems:[
      {t:'Tiled Matrix Multiplication', s:'locked', slug:null},
      {t:'Warp Divergence Analysis',    s:'locked', slug:null},
      {t:'Memory Access Patterns',      s:'locked', slug:null},
      {t:'ncu Profiling',               s:'locked', slug:null},
    ]},
  ]},
];

const KEY = 'cuda-roadmap-v2';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch {}
if (!localStorage.getItem(KEY)) {
  PHASES.forEach(ph => ph.topics.forEach(tg => tg.problems.forEach(p => {
    if (p.s === 'done') saved[p.slug || p.t] = true;
  })));
}
const persist = () => localStorage.setItem(KEY, JSON.stringify(saved));

let currentSlug = null;
let currentChipEl = null;

function updateProgress() {
  const allP = PHASES.flatMap(p => p.topics.flatMap(t => t.problems));
  const total = allP.length;
  const doneCount = allP.filter(p => saved[p.slug || p.t]).length;
  document.getElementById('prog-text').textContent = `${doneCount} / ${total} completed`;
  document.getElementById('prog-fill').style.width = (total ? doneCount/total*100 : 0) + '%';
  PHASES.forEach(p => {
    const problems = p.topics.flatMap(t => t.problems);
    const n = problems.filter(prob => saved[prob.slug || prob.t]).length;
    const el = document.getElementById(`cnt-${p.n}`);
    if (el) el.textContent = `${n} / ${problems.length}`;
  });
}

function toggleDone(key, chipEl) {
  saved[key] = !saved[key];
  chipEl.classList.toggle('done', !!saved[key]);
  chipEl.classList.remove('pop');
  void chipEl.offsetWidth;
  chipEl.classList.add('pop');
  persist();
  updateProgress();
}

function render() {
  const graph = document.getElementById('graph');
  graph.innerHTML = '';

  PHASES.forEach(p => {
    const allProblems = p.topics.flatMap(t => t.problems);
    const sec = document.createElement('div');
    sec.className = 'section';

    const pNode = document.createElement('div');
    pNode.className = 'phase-node';
    pNode.id = `pn-${p.n}`;
    pNode.style.borderColor = p.color + '55';
    pNode.style.background  = p.color + '0e';
    pNode.innerHTML = `
      <div class="pnum" style="color:${p.color}">Phase ${p.n}</div>
      <div class="pname">${p.title}</div>
      <div class="psub" style="color:${p.color}99">${p.sub}</div>
      <div class="pcount" id="cnt-${p.n}">0 / ${allProblems.length}</div>`;
    sec.appendChild(pNode);

    const row = document.createElement('div');
    row.className = 'topic-row';
    row.id = `tr-${p.n}`;

    allProblems.forEach((prob, i) => {
      const key = prob.slug || prob.t;
      const isDone = !!saved[key];
      const hasLesson = !!(prob.slug && LESSONS[prob.slug]);

      const el = document.createElement('div');
      el.className = 'topic' + (isDone ? ' done' : '') + (hasLesson ? ' has-lesson' : '');
      el.id = `tn-${p.n}-${i}`;
      el.dataset.key = key;
      el.dataset.slug = prob.slug || '';
      el.innerHTML = `<div class="dot">✓</div><span>${prob.t}</span>`;

      el.querySelector('.dot').addEventListener('click', e => {
        e.stopPropagation();
        toggleDone(key, el);
      });
      if (hasLesson) {
        el.addEventListener('click', () => openLesson(prob.slug, el));
      }

      row.appendChild(el);
    });

    sec.appendChild(row);
    graph.appendChild(sec);
  });

  const btn = document.createElement('button');
  btn.className = 'reset-btn';
  btn.textContent = '↺ Reset Progress';
  btn.addEventListener('click', () => {
    if (!confirm('Reset all progress?')) return;
    saved = {};
    PHASES.forEach(ph => ph.topics.forEach(tg => tg.problems.forEach(p => {
      if (p.s === 'done') saved[p.slug || p.t] = true;
    })));
    persist();
    document.querySelectorAll('.topic').forEach(el =>
      el.classList.toggle('done', !!saved[el.dataset.key]));
    updateProgress();
    drawLines();
  });
  graph.appendChild(btn);
}

function drawLines() {
  const svg     = document.getElementById('lines');
  const wrapper = document.getElementById('wrapper');
  svg.innerHTML = '';

  const wRect = wrapper.getBoundingClientRect();

  function rr(el) {
    const r = el.getBoundingClientRect();
    return {
      top:    r.top    - wRect.top,
      bottom: r.bottom - wRect.top,
      cx:    (r.left + r.right) / 2 - wRect.left,
    };
  }

  function line(x1,y1,x2,y2,stroke,sw,dash) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','line');
    el.setAttribute('x1',x1); el.setAttribute('y1',y1);
    el.setAttribute('x2',x2); el.setAttribute('y2',y2);
    el.setAttribute('stroke',stroke); el.setAttribute('stroke-width',sw||2);
    if (dash) el.setAttribute('stroke-dasharray',dash);
    svg.appendChild(el);
  }

  function bezier(x1,y1,x2,y2,stroke,sw) {
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    const my = (y1+y2)/2;
    path.setAttribute('d',`M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`);
    path.setAttribute('fill','none');
    path.setAttribute('stroke',stroke);
    path.setAttribute('stroke-width',sw||1.5);
    svg.appendChild(path);
  }

  PHASES.forEach((p, pi) => {
    const phaseEl = document.getElementById(`pn-${p.n}`);
    const rowEl   = document.getElementById(`tr-${p.n}`);
    if (!phaseEl || !rowEl) return;

    const pR = rr(phaseEl);
    const junctionY = pR.bottom + 18;
    line(pR.cx, pR.bottom, pR.cx, junctionY, p.color + '55', 2);

    rowEl.querySelectorAll('.topic').forEach(topic => {
      const tR = rr(topic);
      bezier(pR.cx, junctionY, tR.cx, tR.top, p.color + '45', 1.5);
    });

    if (pi < PHASES.length - 1) {
      const nextEl = document.getElementById(`pn-${PHASES[pi+1].n}`);
      if (nextEl) {
        const nR   = rr(nextEl);
        const rowR = rr(rowEl);
        line(pR.cx, rowR.bottom + 8, nR.cx, nR.top, '#334155', 2, '6,5');
      }
    }
  });

  svg.setAttribute('height', wrapper.scrollHeight);
}

function highlight(code) {
  code = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  code = code.replace(/(\/\/[^\n]*)/g,'<span class="cm">$1</span>');
  code = code.replace(/^(#\w+)/gm,'<span class="pp">$1</span>');
  code = code.replace(/\b(__global__|__device__|__host__|__shared__|__syncthreads|__forceinline__)\b/g,'<span class="deco">$1</span>');
  code = code.replace(/\b(float|int|unsigned|long|void|bool|char|double|size_t|dim3|cudaError_t)\b/g,'<span class="ty">$1</span>');
  code = code.replace(/\b(cudaMalloc|cudaFree|cudaMemcpy|cudaDeviceSynchronize|cudaGetLastError|atomicAdd|blockIdx|threadIdx|blockDim|gridDim)\b/g,'<span class="fn">$1</span>');
  code = code.replace(/\b(if|else|for|while|return|const|static|struct|typedef|do|break|continue|nullptr|NULL)\b/g,'<span class="kw">$1</span>');
  code = code.replace(/\b(\d+\.\d*f?|\d+f?)\b/g,'<span class="nu">$1</span>');
  return code;
}

function openLesson(slug, chipEl) {
  const lesson = LESSONS[slug];
  if (!lesson) return;

  if (currentChipEl) currentChipEl.classList.remove('selected');
  currentSlug   = slug;
  currentChipEl = chipEl;
  if (chipEl) chipEl.classList.add('selected');

  document.getElementById('lp-title').textContent        = lesson.title;
  document.getElementById('lp-section').textContent      = lesson.section;
  document.getElementById('lp-topic').textContent        = lesson.topic;
  document.getElementById('lp-theory-title').textContent = lesson.title;
  document.getElementById('lp-link').href                = lesson.url;
  document.getElementById('lp-filename').textContent     = slug + '.cu';

  function formatDescription(text) {
    return text.split(/\n{2,}|(?<=[.!?])\s{2,}(?=[A-Z])/)
      .map(p => p.trim()).filter(p => p.length > 8).slice(0, 18)
      .map(chunk => {
        if (/^Definition[:\s]/i.test(chunk)) return `<div class="def-box">${chunk}</div>`;
        if (/^Example[:\s]/i.test(chunk))    return `<div class="ex-box">${chunk}</div>`;
        return `<p>${chunk}</p>`;
      }).join('');
  }
  document.getElementById('lp-description').innerHTML = formatDescription(lesson.description);
  document.getElementById('lp-code').innerHTML = highlight(lesson.template);

  const isDone = !!saved[slug];
  const btn = document.getElementById('done-btn');
  btn.textContent = isDone ? '✓ Already Done' : '✓ Mark as Done';
  btn.className   = 'done-btn' + (isDone ? ' already-done' : '');

  const copyBtn = document.getElementById('copy-btn');
  copyBtn.textContent = 'Copy'; copyBtn.className = 'copy-btn';

  document.getElementById('app-body').classList.add('panel-open');
  switchTab('theory');

  const vizHtml   = document.getElementById('lp-viz-html');
  const vizCanvas = document.getElementById('viz-canvas');
  if (lesson.html_content) {
    vizHtml.innerHTML = lesson.html_content;
    vizHtml.style.display = 'block';
    vizCanvas.style.display = 'none';
  } else if (VISUALIZATIONS && VISUALIZATIONS[slug]) {
    vizHtml.style.display = 'none';
    vizCanvas.style.display = 'block';
    const ctx = vizCanvas.getContext('2d');
    ctx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);
    try { VISUALIZATIONS[slug](vizCanvas, vizCanvas.width, vizCanvas.height, ctx); }
    catch(e) { console.warn('viz error', slug, e); }
  } else {
    vizHtml.style.display = 'none';
    vizCanvas.style.display = 'none';
  }
}

function closePanel() {
  document.getElementById('app-body').classList.remove('panel-open');
  if (currentChipEl) currentChipEl.classList.remove('selected');
  currentSlug = null; currentChipEl = null;
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', ['theory','code'][i] === name));
  document.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('active', c.id === 'tab-'+name));
}

function copyCode() {
  navigator.clipboard.writeText(document.getElementById('lp-code').innerText).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!'; btn.className = 'copy-btn copied';
    setTimeout(() => { btn.textContent='Copy'; btn.className='copy-btn'; }, 2000);
  });
}

function markDone() {
  if (!currentSlug || saved[currentSlug]) return;
  saved[currentSlug] = true;
  persist();
  const btn = document.getElementById('done-btn');
  btn.textContent = '✓ Already Done'; btn.className = 'done-btn already-done';
  const chip = document.querySelector(`[data-slug="${currentSlug}"]`);
  if (chip) { chip.classList.add('done'); currentChipEl = chip; }
  updateProgress();
}

VISUALIZATIONS_PLACEHOLDER
render();
updateProgress();
requestAnimationFrame(() => requestAnimationFrame(drawLines));
window.addEventListener('resize', () => requestAnimationFrame(drawLines));
document.getElementById('lesson-pane').addEventListener('transitionend', drawLines);
</script>
</body>
</html>
"""

# Inject lessons data and visualizations
viz_js = build_viz_js()
final_html = HTML.replace('LESSONS_PLACEHOLDER', lessons_js)
final_html = final_html.replace('VISUALIZATIONS_PLACEHOLDER', viz_js)

OUT_FILE.write_text(final_html, encoding='utf-8')
print(f'Generated: {OUT_FILE}  ({len(final_html)//1024}KB)')
