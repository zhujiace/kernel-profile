import numpy as np
import os

def read_binary(filename, size):
    return np.fromfile(filename, dtype=np.float32, count=size)

def compare_outputs(output_file, ref_file, size, tolerance=1e-2):
    if not os.path.exists(output_file) or not os.path.exists(ref_file):
        return False
    output = read_binary(output_file, size)
    reference = read_binary(ref_file, size)
    if output.shape != reference.shape:
        return False
    diff = np.abs(output - reference)
    return np.all(diff < tolerance)

if __name__ == "__main__":
    N = 128
    K_ROWS, K_COLS = 24, 24
    out_rows = N - K_ROWS + 1
    out_cols = N - K_COLS + 1
    size_out = out_rows * out_cols

    out_file = "data/conv_output.bin"
    ref_file = "data/conv_ref.bin"

    if compare_outputs(out_file, ref_file, size_out):
        print("T")
    else:
        print("F")
