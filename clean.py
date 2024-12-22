import gc
import torch

# torch.cuda.empty_cache()
# gc.collect()

import torch
import torch.distributed as dist

# dist.init_process_group(backend='nccl', init_method='env://')

print(f"PyTorch version: {torch.__version__}") # 1.12.0
print(f"CUDA version: {torch.version.cuda}") # 11.6
print(f"NCCL version: {torch.cuda.nccl.version()}") # 2,10,3