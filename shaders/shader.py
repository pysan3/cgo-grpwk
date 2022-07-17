from sys import setdlopenflags
import warnings

import torch
import torch.nn as nn

from pytorch3d.renderer.blending import (
    BlendParams,
    softmax_rgb_blend,
    hard_rgb_blend,
)


from pytorch3d.renderer.lighting import (
    PointLights,
)

from .shading import (
    prewitt_shading,
    world_normal_shading,
)


class WorldNormalShader(nn.Module):
    """
    ああああああ！！！！
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(
            device=device)
        # self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                    or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = world_normal_shading(
            meshes=meshes,
            fragments=fragments,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


class WorldDepthShader(nn.Module):
    """
    ああああああ！！！！
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(
            device=device)
        # self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                    or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        blend_params = kwargs.get("blend_params", self.blend_params)
        h = fragments.zbuf.shape[1]
        w = fragments.zbuf.shape[2]
        pp = fragments.zbuf.shape[3]
        zbuf = fragments.zbuf.view(pp, 1, h, w)
        colors = zbuf[pp - 1].view(1, 1, h, w)
        return colors


class OutlineShader(nn.Module):
    """
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None, power=3, offset=1.0
    ):
        super().__init__()
        # self.lights = lights if lights is not None else PointLights(device=device)
        # self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.power = power
        self.offset = offset

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        # self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                    or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = prewitt_shading(
            meshes=meshes,
            fragments=fragments,
            power=self.power,
            offset=self.offset
        )

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        #images = softmax_rgb_blend(colors, fragments, blend_params, znear=znear, zfar=zfar)
        return colors


class SimpleShader(nn.Module):
    def __init__(self, device="cpu",  cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image