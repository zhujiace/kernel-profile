import numpy as np
import os

N = 1 << 10
np.random.seed(49)

preds = (np.random.randn(N) * 10).astype(np.float32)
targets = (np.random.randn(N) * 10).astype(np.float32)
mse = np.mean((preds - targets) ** 2).astype(np.float32)

os.makedirs("data", exist_ok=True)
preds.tofile("data/mse_preds.bin")
targets.tofile("data/mse_targets.bin")
np.array([mse], dtype=np.float32).tofile("data/mse_ref.bin")
