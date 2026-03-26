import torch


def _normalize_dim(dim, ndim):
    return dim if dim >= 0 else ndim + dim


def _expand_index(index, src, dim):
    expanded_index = index.long()
    while expanded_index.dim() < src.dim():
        expanded_index = expanded_index.unsqueeze(-1)
    return expanded_index.expand_as(src)


def scatter_sum(src, index, dim=-1, dim_size=None):
    dim = _normalize_dim(dim, src.dim())
    expanded_index = _expand_index(index, src, dim)
    if dim_size is None:
        dim_size = int(expanded_index.max().item()) + 1 if expanded_index.numel() > 0 else 0

    output_shape = list(src.shape)
    output_shape[dim] = dim_size
    out = torch.zeros(output_shape, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, expanded_index, src)
    return out


def scatter_softmax(src, index, dim=-1):
    dim = _normalize_dim(dim, src.dim())
    expanded_index = _expand_index(index, src, dim)
    dim_size = int(expanded_index.max().item()) + 1 if expanded_index.numel() > 0 else 0

    reduce_shape = list(src.shape)
    reduce_shape[dim] = dim_size
    max_values = torch.full(reduce_shape, float("-inf"), dtype=src.dtype, device=src.device)
    max_values.scatter_reduce_(dim, expanded_index, src, reduce="amax", include_self=True)
    gathered_max = torch.gather(max_values, dim, expanded_index)

    shifted = src - gathered_max
    exp_values = torch.exp(shifted)
    normalizers = torch.zeros(reduce_shape, dtype=src.dtype, device=src.device)
    normalizers.scatter_add_(dim, expanded_index, exp_values)
    gathered_normalizers = torch.gather(normalizers, dim, expanded_index).clamp_min_(1e-12)
    return exp_values / gathered_normalizers
