import torch
from dataclasses import dataclass

from tutorials.utils.generate_cow_renders import generate_cow_renders
from io import BytesIO 
from PIL import Image
import numpy as np

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


def load_images(num_of_images: int, root_dir: str)->list[np.ndarray]:
    images = []
    for i in range(num_of_images):
        image = Image.open(f'{root_dir}{i}.png')
        image = np.array(image)
        images.append(image)
    return images

def copy_images_to_tensor(images: list)->torch.Tensor:
    image_shape = images[0].shape
    tensor = torch.zeros(len(images), image_shape[0], image_shape[1], 3)
    for i, image in enumerate(images):
        tensor[i] = torch.from_numpy(image/255.0)
    return tensor

def generate_data_from_files(num_of_images: int, root_dir: str, device=None):
    images = load_images(num_of_images, root_dir)
    tensor = copy_images_to_tensor(images)
    if device is not None:
        tensor = tensor.to(device)
    target_cameras, *_ = generate_cow_renders(num_views=num_of_images)
    # シルエットは多分使わないので便宜上RGB画像を使う
    return target_cameras, ImageDatas(tensor, tensor)