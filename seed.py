# Set seeds for Python random number generator
import os
import random
import numpy as np
import torch

seed = 1    #42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Set this to False for fully deterministic results
torch.use_deterministic_algorithms(True)

cuda_version = torch.version.cuda
if cuda_version is not None:
    if cuda_version >= "10.2":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
