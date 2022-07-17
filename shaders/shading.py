import torch
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes


def world_normal_shading(meshes, fragments) -> torch.Tensor:
    """
    write a comments!!
    """

    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    return pixel_normals


def prewitt_shading(meshes, fragments, power, offset) -> torch.Tensor:
    """
    """
    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    h = fragments.zbuf.shape[1]
    w = fragments.zbuf.shape[2]
    pp = fragments.zbuf.shape[3]
    zbuf = fragments.zbuf.view(pp, 1, h, w)
    zbuf = torch.clamp(zbuf, min=1)
    zbuf = torch.pow(zbuf, power)
    pixel_depth = []
    for i in range(pp):
        pixel_depth.append(zbuf[i].view(1, 1, h, w))

    prewitt_h = torch.cuda.FloatTensor(
        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    ).view(1, 1, 3, 3)

    prewitt_v = torch.cuda.FloatTensor(
        [[-1, -1, -1],
         [0, 0, 0],
         [1, 1, 1]]
    ).view(1, 1, 3, 3)

    l = []
    for filter_input in pixel_depth:
        pixel_h = F.conv2d(
            input=filter_input,
            weight=prewitt_h,
            stride=1,
            padding=1
        )
        pixel_v = F.conv2d(
            input=filter_input,
            weight=prewitt_v,
            stride=1,
            padding=1
        )
        filtered_pix = torch.sqrt(
            pixel_h * pixel_h + pixel_v * pixel_v)
        l.append(filtered_pix)

    pixel = torch.stack(
        ([f for f in l]), 0
    ).view(1, h, w, pp)
    offset = torch.clamp(torch.tensor(offset), min=0, max=1)
    pixel = 1 / (1 + torch.exp(-20 * (F.tanh(pixel) - offset)))
    return pixel
