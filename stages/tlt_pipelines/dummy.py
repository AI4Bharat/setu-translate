# import os
# os.environ["XLA_USE_BF16"]="1"

import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend

print(xm.xrt_world_size())

print(xm.xla_device())


def _mp_fn(index):
    pass

if __name__ == "__main__":

    print("fuckerd")

    print(xm.xrt_world_size())

    print("fucked")

    print(xm.xla_device())

    # xmp.spawn(
    #     _mp_fn,
    #     args=(),
    #     start_method='fork'
    # )
