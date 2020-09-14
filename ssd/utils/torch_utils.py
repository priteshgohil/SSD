import time

import torch

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()