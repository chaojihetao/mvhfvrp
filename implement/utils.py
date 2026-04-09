import torch
import random
from einops import rearrange
from torch import Tensor
from tensordict import TensorDict, stack
from typing import Dict, List, Any, Tuple
import concurrent.futures
    

def get_distance(x: Tensor,
                 y: Tensor) -> Tensor:
    """
        Calculates Euclidean distance or retrieves distance from a precomputed matrix.

    Args:
        x: First tensor. Can be coordinates of shape `[..., dim]` or indices of shape `[...]`.
        y: Second tensor. Can be coordinates of shape `[..., dim]` or indices of shape `[...]`.
    """
    
    # Calculate Euclidean distance if no matrix is provided
    distance = (x - y).norm(p=2, dim=-1)    # [B, N, 1, 2] , [B, 1, M, 2] -> [B, N, M]

    return distance


def get_distance_by_matrix(distance_matrix: Tensor,
                           x_idx: Tensor,
                           y_idx: Tensor) -> Tensor:
    """
        Retrieves distances from a precomputed distance matrix.
        Given a distance matrix of shape [B, N, N], x_idx of shape [B, R] and y_idx of shape [B, M],
        it returns a tensor of shape [B, R, M] where the entry [b, i, j] is distance_matrix[b, x_idx[b, i], y_idx[b, j]].
    """
    # Handle the simple case of 1D indices for efficiency
    if x_idx.dim() == 1 and y_idx.dim() == 1:
        batch_size = distance_matrix.size(0)
        # Create batch indices [0, 1, ..., B-1]
        batch_idx = torch.arange(batch_size, device=distance_matrix.device)
        # Use advanced indexing to get the values directly
        return distance_matrix[batch_idx, x_idx, y_idx]

    batch_size, R_size = x_idx.shape
    _batch_size, M_size = y_idx.shape
    
    # Gather rows
    # x_idx: [B, R] -> [B, R, N]
    x_idx_expanded = x_idx.unsqueeze(-1).expand(batch_size, R_size, distance_matrix.size(-1))
    # gathered_rows: [B, R, N]
    gathered_rows = distance_matrix.gather(1, x_idx_expanded)

    # Gather columns
    # y_idx: [B, M] -> [B, R, M]
    y_idx_expanded = y_idx.unsqueeze(1).expand(batch_size, R_size, M_size)
    # result: [B, R, M]
    result = gathered_rows.gather(2, y_idx_expanded)
    return result

def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)