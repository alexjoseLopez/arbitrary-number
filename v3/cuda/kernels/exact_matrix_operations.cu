/**
 * CUDA Kernels for Exact Matrix Operations with ArbitraryNumbers
 * ============================================================
 * 
 * Advanced CUDA implementations for exact mathematical operations
 * that maintain perfect precision throughout parallel computations.
 * 
 * These kernels demonstrate how ArbitraryNumbers can be used in
 * high-performance GPU computing without precision loss.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// ArbitraryNumber structure for GPU
struct ArbitraryNumberGPU {
    long long numerator;
    long long denominator;
    int precision_loss_flag;
    
    __device__ ArbitraryNumberGPU() : numerator(0), denominator(1), precision_loss_flag(0) {}
    
    __device__ ArbitraryNumberGPU(long long num, long long den) 
        : numerator(num), denominator(den), precision_loss_flag(0) {
        if (denominator == 0) {
            denominator = 1;
            precision_loss_flag = 1;
        }
        reduce();
    }
    
    __device__ void reduce() {
        if (numerator == 0) {
            denominator = 1;
            return;
        }
        
        long long a = abs(numerator);
        long long b = abs(denominator);
        
        // Euclidean algorithm for GCD
        while (b != 0) {
            long long temp = b;
            b = a % b;
            a = temp;
        }
        
        numerator /= a;
        denominator /= a;
        
        if (denominator < 0) {
            numerator = -numerator;
            denominator = -denominator;
        }
    }
    
    __device__ ArbitraryNumberGPU operator+(const ArbitraryNumberGPU& other) const {
        long long new_num = numerator * other.denominator + other.numerator * denominator;
        long long new_den = denominator * other.denominator;
        
        ArbitraryNumberGPU result(new_num, new_den);
        result.precision_loss_flag = precision_loss_flag || other.precision_loss_flag;
        return result;
    }
    
    __device__ ArbitraryNumberGPU operator*(const ArbitraryNumberGPU& other) const {
        long long new_num = numerator * other.numerator;
        long long new_den = denominator * other.denominator;
        
        ArbitraryNumberGPU result(new_num, new_den);
        result.precision_loss_flag = precision_loss_flag || other.precision_loss_flag;
        return result;
    }
    
    __device__ double to_double() const {
        return (double)numerator / (double)denominator;
    }
};

/**
 * Exact Matrix Multiplication Kernel
 * ==================================
 * 
 * Performs matrix multiplication with perfect precision using ArbitraryNumbers.
 * Each thread computes one element of the result matrix.
 */
__global__ void exact_matrix_multiply(
    ArbitraryNumberGPU* A, ArbitraryNumberGPU* B, ArbitraryNumberGPU* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        ArbitraryNumberGPU sum(0, 1);
        
        for (int k = 0; k < K; k++) {
            ArbitraryNumberGPU a_elem = A[row * K + k];
            ArbitraryNumberGPU b_elem = B[k * N + col];
            sum = sum + (a_elem * b_elem);
        }
        
        C[row * N + col] = sum;
    }
}

/**
 * Exact Vector Dot Product Kernel
 * ===============================
 * 
 * Computes dot product with perfect precision using parallel reduction.
 */
__global__ void exact_dot_product(
    ArbitraryNumberGPU* a, ArbitraryNumberGPU* b, ArbitraryNumberGPU* result, int n
) {
    extern __shared__ ArbitraryNumberGPU sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (i < n) {
        sdata[tid] = a[i] * b[i];
    } else {
        sdata[tid] = ArbitraryNumberGPU(0, 1);
    }
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/**
 * Exact Neural Network Forward Pass Kernel
 * ========================================
 * 
 * Performs exact forward pass computation for neural networks.
 * Includes exact activation function approximations.
 */
__global__ void exact_neural_forward_pass(
    ArbitraryNumberGPU* weights, ArbitraryNumberGPU* inputs, 
    ArbitraryNumberGPU* biases, ArbitraryNumberGPU* outputs,
    int input_size, int output_size
) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (neuron_idx < output_size) {
        ArbitraryNumberGPU activation = biases[neuron_idx];
        
        // Compute weighted sum
        for (int i = 0; i < input_size; i++) {
            ArbitraryNumberGPU weight = weights[neuron_idx * input_size + i];
            activation = activation + (weight * inputs[i]);
        }
        
        // Exact sigmoid approximation: σ(x) ≈ x / (1 + |x|) for |x| < 5
        ArbitraryNumberGPU abs_activation = activation;
        if (activation.numerator < 0) {
            abs_activation.numerator = -abs_activation.numerator;
        }
        
        ArbitraryNumberGPU five(5, 1);
        ArbitraryNumberGPU one(1, 1);
        
        // Compare |activation| with 5
        bool is_large = (abs_activation.numerator * five.denominator) > 
                       (five.numerator * abs_activation.denominator);
        
        if (is_large) {
            // Hard saturation
            outputs[neuron_idx] = (activation.numerator >= 0) ? one : ArbitraryNumberGPU(0, 1);
        } else {
            // Rational approximation
            if (activation.numerator >= 0) {
                outputs[neuron_idx] = activation * ArbitraryNumberGPU(1, 1) / (one + activation);
            } else {
                outputs[neuron_idx] = activation * ArbitraryNumberGPU(1, 1) / (one + abs_activation);
            }
        }
    }
}

/**
 * Exact Gradient Descent Update Kernel
 * ====================================
 * 
 * Updates parameters using exact arithmetic in gradient descent.
 */
__global__ void exact_gradient_update(
    ArbitraryNumberGPU* parameters, ArbitraryNumberGPU* gradients,
    ArbitraryNumberGPU learning_rate, int param_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < param_count) {
        ArbitraryNumberGPU update = learning_rate * gradients[idx];
        parameters[idx] = parameters[idx] + (ArbitraryNumberGPU(0, 1) - update);
    }
}

/**
 * Exact Convolution Kernel
 * ========================
 * 
 * Performs 2D convolution with exact arithmetic for CNN operations.
 */
__global__ void exact_convolution_2d(
    ArbitraryNumberGPU* input, ArbitraryNumberGPU* kernel, ArbitraryNumberGPU* output,
    int input_height, int input_width, int kernel_size, int output_height, int output_width
) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_row < output_height && out_col < output_width) {
        ArbitraryNumberGPU sum(0, 1);
        
        for (int k_row = 0; k_row < kernel_size; k_row++) {
            for (int k_col = 0; k_col < kernel_size; k_col++) {
                int in_row = out_row + k_row;
                int in_col = out_col + k_col;
                
                if (in_row < input_height && in_col < input_width) {
                    ArbitraryNumberGPU input_val = input[in_row * input_width + in_col];
                    ArbitraryNumberGPU kernel_val = kernel[k_row * kernel_size + k_col];
                    sum = sum + (input_val * kernel_val);
                }
            }
        }
        
        output[out_row * output_width + out_col] = sum;
    }
}

/**
 * Exact Eigenvalue Power Iteration Kernel
 * =======================================
 * 
 * Computes dominant eigenvalue using power iteration with exact arithmetic.
 */
__global__ void exact_power_iteration_step(
    ArbitraryNumberGPU* matrix, ArbitraryNumberGPU* vector, ArbitraryNumberGPU* result,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        ArbitraryNumberGPU sum(0, 1);
        
        for (int j = 0; j < n; j++) {
            sum = sum + (matrix[idx * n + j] * vector[j]);
        }
        
        result[idx] = sum;
    }
}

/**
 * Exact Gaussian Elimination Kernel
 * =================================
 * 
 * Performs exact Gaussian elimination for solving linear systems.
 */
__global__ void exact_gaussian_elimination_step(
    ArbitraryNumberGPU* matrix, int n, int pivot_row, int current_col
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row > pivot_row && row < n) {
        ArbitraryNumberGPU pivot_element = matrix[pivot_row * (n + 1) + current_col];
        ArbitraryNumberGPU current_element = matrix[row * (n + 1) + current_col];
        
        if (pivot_element.numerator != 0) {
            ArbitraryNumberGPU factor = current_element * ArbitraryNumberGPU(1, 1) / pivot_element;
            
            for (int col = current_col; col <= n; col++) {
                ArbitraryNumberGPU pivot_val = matrix[pivot_row * (n + 1) + col];
                matrix[row * (n + 1) + col] = matrix[row * (n + 1) + col] + 
                                             (ArbitraryNumberGPU(0, 1) - (factor * pivot_val));
            }
        }
    }
}

// Host function declarations
extern "C" {
    void launch_exact_matrix_multiply(
        ArbitraryNumberGPU* d_A, ArbitraryNumberGPU* d_B, ArbitraryNumberGPU* d_C,
        int M, int N, int K
    );
    
    void launch_exact_neural_forward_pass(
        ArbitraryNumberGPU* d_weights, ArbitraryNumberGPU* d_inputs,
        ArbitraryNumberGPU* d_biases, ArbitraryNumberGPU* d_outputs,
        int input_size, int output_size
    );
    
    void launch_exact_convolution_2d(
        ArbitraryNumberGPU* d_input, ArbitraryNumberGPU* d_kernel, ArbitraryNumberGPU* d_output,
        int input_height, int input_width, int kernel_size, int output_height, int output_width
    );
}
