import torch


def huber(x: torch.Tensor, y: torch.Tensor, scaling: float = 0.1) -> torch.Tensor:
    """
    A helper function for evaluating the smooth L1 (huber) loss between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss
