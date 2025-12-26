#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <fstream>
#include <string>

const int N = 128;
const int K_ROWS = 24;
const int K_COLS = 24;

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int in_rows, int in_cols, int k_rows, int k_cols) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_rows = in_rows - k_rows + 1;
    int out_cols = in_cols - k_cols + 1;

    if (out_row >= out_rows || out_col >= out_cols) return;

    float acc = 0.0f;
    for (int i = 0; i < k_rows; ++i) {
        for (int j = 0; j < k_cols; ++j) {
            int in_r = out_row + i;
            int in_c = out_col + j;
            acc += input[in_r * in_cols + in_c] * kernel[i * k_cols + j];
        }
    }

    output[out_row * out_cols + out_col] = acc;
}

void read_binary_float(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void write_binary(const std::string& filename, const float* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot write: " << filename << std::endl;
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    out.close();
}

int main() {
    int in_rows = N;
    int in_cols = N;
    int out_rows = in_rows - K_ROWS + 1;
    int out_cols = in_cols - K_COLS + 1;

    size_t in_size = in_rows * in_cols;
    size_t k_size = K_ROWS * K_COLS;
    size_t out_size = out_rows * out_cols;

    float* h_input = new float[in_size];
    float* h_kernel = new float[k_size];
    float* h_output = new float[out_size];

    read_binary_float("data/conv_input.bin", h_input, in_size);
    read_binary_float("data/conv_kernel.bin", h_kernel, k_size);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, in_size * sizeof(float));
    cudaMalloc(&d_kernel, k_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_input, h_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, k_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((out_cols + threads.x - 1) / threads.x,
                (out_rows + threads.y - 1) / threads.y);

    conv2d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output, in_rows, in_cols, K_ROWS, K_COLS);
    cudaMemcpy(h_output, d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/conv_output.bin", h_output, out_size);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}
