import random

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
