from absl import flags
import torch
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size: list[int] = [64] * 3, voxel_size: int = 0.1):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # After evaluating torch.sigmoid(self.log_colors), we get
        # a neutral gray color everywhere.
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer

    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)

        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities=densities[None].expand(
                batch_size, *self.log_densities.shape),
            features=colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )

        # Given cameras and volumes, run the renderer
        # and return only the first output value
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]


def get_model(opts: flags.FLAGS, data_shape: int, device):
    # render_size describes the size of both sides of the
    # rendered images in pixels. We set this to the same size
    # as the target images. I.e. we render at the same
    # size as the ground truth images.
    renderer = VolumeRenderer(
        raysampler=NDCMultinomialRaysampler(
            image_width=data_shape,
            image_height=data_shape,
            n_pts_per_ray=150,
            min_depth=0.1,
            max_depth=opts.vol_extent_world,
        ),
        raymarcher=EmissionAbsorptionRaymarcher()
    )

    volume_model = VolumeModel(
        renderer,
        volume_size=[opts.vol_size] * 3,
        voxel_size=opts.vol_extent_world / opts.vol_size,
    ).to(device)

    return volume_model
