import torch
from dataclasses import dataclass

from tutorials.utils.generate_cow_renders import generate_cow_renders


@dataclass
class ImageDatas:
    images: torch.Tensor
    silhouettes: torch.Tensor

    @property
    def shape(self) -> int:
        return self.images.shape[1]

    def __len__(self):
        return len(self.images)


def tutorial_generate_cow_renders(num_views: int = 40, device=None):
    target_cameras, *_data = generate_cow_renders(num_views=num_views)
    if device is not None:
        _data = (d.to(device) for d in _data)
    target_data = ImageDatas(*_data)
    return target_cameras, target_data
