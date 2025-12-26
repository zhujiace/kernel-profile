#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <string>

__global__ void mseKernel(const float* predictions, const float* targets, size_t numElements, float* sum) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        float diff = predictions[idx] - targets[idx];
        float sq_diff = diff * diff;
        atomicAdd(sum, sq_diff);
    }
}

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void write_binary(const std::string& filename, const float* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot write file: " << filename << std::endl;
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    out.close();
}

int main() {
    size_t N = 1 << 10;
    size_t input_size = N * sizeof(float);

    std::string pred_file = "data/mse_preds.bin";
    std::string target_file = "data/mse_targets.bin";

    float* h_preds = (float*)malloc(input_size);
    float* h_targets = (float*)malloc(input_size);

    read_binary(pred_file, h_preds, N);
    read_binary(target_file, h_targets, N);

    float *d_preds, *d_targets, *d_sum;
    cudaMalloc(&d_preds, input_size);
    cudaMalloc(&d_targets, input_size);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_preds, h_preds, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mseKernel<<<blocks, threads>>>(d_preds, d_targets, N, d_sum);

    float h_sum = 0.0f;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float mse = h_sum / N;

    // 写结果
    write_binary("data/mse_out.bin", &mse, 1);

    cudaFree(d_preds);
    cudaFree(d_targets);
    cudaFree(d_sum);
    free(h_preds);
    free(h_targets);

    return 0;
}
