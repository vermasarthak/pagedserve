import torch

from pagedserve.model.tensor_parallel import TensorParallelConfig, TensorParallelGroup


def test_tensor_parallel_config():
    cfg = TensorParallelConfig(world_size=4, rank=1)
    group = TensorParallelGroup(cfg)

    local_heads, start, end = group.split_heads(32)
    assert local_heads == 8
    assert start == 8
    assert end == 16


def test_all_reduce():
    cfg = TensorParallelConfig(world_size=2, rank=0)
    group = TensorParallelGroup(cfg)
    t = torch.ones(2, 4)
    res = group.all_reduce(t)
    assert res.shape == (2, 4)
