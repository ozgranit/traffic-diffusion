# Set seeds for Python random number generator
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Set this to False for fully deterministic results
torch.use_deterministic_algorithms(True)
