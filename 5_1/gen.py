import numpy as np
import os

np.random.seed(42)

N = 128 
k_rows, k_cols = 24, 24

def conv2d_valid(input_mat, kernel):
    in_rows, in_cols = input_mat.shape
    k_rows, k_cols = kernel.shape
    out_rows = in_rows - k_rows + 1
    out_cols = in_cols - k_cols + 1
    output = np.zeros((out_rows, out_cols), dtype=np.float32)

    for i in range(out_rows):
        for j in range(out_cols):
            patch = input_mat[i:i+k_rows, j:j+k_cols]
            output[i, j] = np.sum(patch * kernel)
    return output

os.makedirs("data", exist_ok=True)

input_mat = np.random.randn(N, N).astype(np.float32)
kernel = np.random.randn(k_rows, k_cols).astype(np.float32)
output = conv2d_valid(input_mat, kernel)

input_mat.ravel().tofile("data/conv_input.bin")
kernel.ravel().tofile("data/conv_kernel.bin")
output.ravel().tofile("data/conv_ref.bin")
