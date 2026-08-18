import os
import torch

# Intel CPU optimization
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

torch.set_num_threads(8)