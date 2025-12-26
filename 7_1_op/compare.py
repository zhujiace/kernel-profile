import numpy as np
import os

def read_binary(filename, size):
    return np.fromfile(filename, dtype=np.float32, count=size)

def compare_scalar(output_file, ref_file, tol=1e-1):
    if not os.path.exists(output_file) or not os.path.exists(ref_file):
        return False
    out = read_binary(output_file, 1)[0]
    ref = read_binary(ref_file, 1)[0]
    return abs(out - ref) < tol

if __name__ == "__main__":
    out_file = "data/mse_out.bin"
    ref_file = "data/mse_ref.bin"

    if compare_scalar(out_file, ref_file):
        print("T")
    else:
        print("F")
