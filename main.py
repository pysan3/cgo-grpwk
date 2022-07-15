
# %%
import os
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
from rich.traceback import install
from absl import app, flags
import torch
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import FoVPerspectiveCameras

from tutorials.utils.plot_image_grid import image_grid
from utils.eval_func import huber
from utils.setup_env import print_torch, check_cuda
from utils.model_utils import ModelLoader
from utils.data_loader import ImageDatas, tutorial_generate_cow_renders, generate_data_from_files
from model_define import get_model

install()  # Fancier traceback from rich library
opts = flags.FLAGS  # parse command line args with Abseil-py

flags.DEFINE_string('project_name', 'cgo-grpwk', 'Name of this project. Do not change this from default value.')
# runtime config opts
flags.DEFINE_bool('verbose', True, 'Print more detailed informations during training.')
flags.DEFINE_bool('vis', False, 'Toggle to visualize the rendered result during training')

flags.DEFINE_string('model_out', './out', 'Path to output model and training progress')
flags.DEFINE_integer('model_save_every', 10, 'Interval of saving current model weights')
flags.DEFINE_integer('model_load_iter', 0, 'Load model at this iteration. (-1) means the latest')

# model param opts
flags.DEFINE_integer('vol_size', 128, 'Size of volume to express the space')
flags.DEFINE_float(
    'vol_extent_world',
    3.0,
    'Our rendered scene is centered around (0,0,0) and is enclosed inside a bounding box'
)

# training opts
flags.DEFINE_string('optim', 'Adam', 'Name of optimizer. Will load as `getattr(torch.optim, opts.optim)`')
flags.DEFINE_float('lr', 0.1, 'learning rate')
flags.DEFINE_integer('batch_size', 10, 'Size of batch while training')
flags.DEFINE_integer('num_iters', 300, 'How many iterations for training')

print_torch()
if not check_cuda():
    raise Exception('This code will fail without CUDA. Please install it and rerun again.')
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    opts: flags.FlagValues,
    model: ModelLoader,
    target_data: ImageDatas,
    target_cameras: FoVPerspectiveCameras,
    start_iter: int = 1,
):
    loss_hist = []
    for iteration in track(range(max(start_iter, opts.model_load_iter), opts.num_iters + 1), description='Training...'):
        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(opts.num_iters * 0.75):
            print('Decreasing learning rate 10-fold ...')
            model.update_optim_lr(opts.lr * 0.1)

        # Zero the optimizer gradient.
        model.optimizer.zero_grad()

        # Sample random batch indices.
        batch_idx = torch.randperm(len(target_cameras))[:opts.batch_size]

        # Sample the minibatch of cameras.
        batch_cameras = FoVPerspectiveCameras(
            R=target_cameras.R[batch_idx],
            T=target_cameras.T[batch_idx],
            znear=target_cameras.znear[batch_idx],  # type: ignore
            zfar=target_cameras.zfar[batch_idx],  # type: ignore
            aspect_ratio=target_cameras.aspect_ratio[batch_idx],  # type: ignore
            fov=target_cameras.fov[batch_idx],  # type: ignore
            device=device,
        )

        # Evaluate the volumetric model.
        rendered_data = ImageDatas(*model(batch_cameras).split([3, 1], dim=-1))

        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # target silhouettes.
        # シルエットは使わない
        # sil_err = huber(rendered_data.silhouettes[..., 0], target_data.silhouettes[batch_idx]).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # target ground truth images.
        color_err = huber(rendered_data.images, target_data.images[batch_idx]).abs().mean()

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss: torch.Tensor = color_err # + sil_err
        loss_hist.append(loss.item())
        model.save_iteration(iteration, {
            'color_err': color_err.item(),
            # 'sil_err': sil_err.item(),
            'loss (color + sil)': loss.item(),
        })

        # Take the optimization step.
        loss.backward()
        model.optimizer.step()

        # Visualize the renders every 40 iterations.
        if iteration % 40 == 0 or iteration == opts.num_iters:
            # Visualize only a single randomly selected element of the batch.
            im_show_idx = int(torch.randint(low=0, high=opts.batch_size, size=(1,)))
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax = ax.ravel()

            def clamp_and_detach(x):
                return x.clamp(0.0, 1.0).cpu().detach().numpy()
            ax[0].imshow(clamp_and_detach(rendered_data.images[im_show_idx]))
            ax[1].imshow(clamp_and_detach(target_data.images[batch_idx[im_show_idx], ..., :3]))
            ax[2].imshow(clamp_and_detach(rendered_data.silhouettes[im_show_idx, ..., 0]))
            ax[3].imshow(clamp_and_detach(target_data.silhouettes[batch_idx[im_show_idx]]))
            axis_names = ("rendered image", "target image", "rendered silhouette", "target silhouette")
            for ax_, title_ in zip(ax, axis_names):
                ax_.grid("off")
                ax_.axis("off")
                ax_.set_title(title_)
            fig.canvas.draw()
            img_out_dir = model.output_dir / 'images'
            img_out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(img_out_dir / f'{iteration:04}.png')
            if opts.vis:
                fig.show()

    return loss_hist


def generate_rotating_volume(volume_model: ModelLoader, target_cameras: FoVPerspectiveCameras, n_frames=50):
    print('[green]Eval ...[/]')
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    for R, T in zip(track(Rs, description='Generating rotating volume...'), Ts):
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            znear=target_cameras.znear[0],  # type: ignore
            zfar=target_cameras.zfar[0],  # type: ignore
            aspect_ratio=target_cameras.aspect_ratio[0],  # type: ignore
            fov=target_cameras.fov[0],  # type: ignore
            device=device,
        )
        frames.append(volume_model(camera)[..., :3].clamp(0.0, 1.0))
    return torch.cat(frames)


def main(_):
    image_dir = 'tutorials/out/'
    num_of_images = sum(os.path.isfile(os.path.join(image_dir, name)) for name in os.listdir(image_dir))
    target_cameras, target_data = generate_data_from_files(num_of_images=num_of_images, root_dir=image_dir, device=device) #tutorial_generate_cow_renders(num_views=40, device=device)
    if opts.verbose:
        print(f'Generated {len(target_data)} images/silhouettes/cameras.')

    prev_model_iter = 1
    model = ModelLoader(opts, get_model(opts, target_data.shape, device))
    if model.prev_model_path is not None:
        prev_model_iter = opts.model_load_iter if opts.model_load_iter > 0 else model.get_latest_iteration()
        print(f'Found pre trained model. Start training from what iter? (default: {prev_model_iter})')
        try:
            prev_model_iter = int(input('> '))
        except Exception as e:
            print(f'Could not parse your input {e=}. Starting from {prev_model_iter}')
    train(opts, model, target_data, target_cameras, start_iter=prev_model_iter)

    with torch.no_grad():
        rotating_volume_frames = generate_rotating_volume(model, target_cameras, n_frames=7 * 4)

    image_grid(rotating_volume_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=7, rgb=True, fill=True)
    plt.savefig(model.output_dir / 'eval.png')
    print(f'[green]Open {model.output_dir / "eval.png"} for results.[/]')
    if opts.vis:
        plt.show()


if __name__ == "__main__":
    app.run(main)

# %%
