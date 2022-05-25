from absl import flags
import re
from rich import print
from typing import Dict, Any
from pathlib import Path
import torch
from torch.utils.tensorboard.writer import SummaryWriter


class ModelLoader:
    def __init__(self, opts: flags.FLAGS, model: torch.nn.Module):
        self.opts = opts
        self.output_dir = Path(opts.model_out)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(opts.model_save_every)

        self.model = model
        self.prev_model_path = self._load_prev_model(opts.model_load_iter)
        if self.prev_model_path is not None:
            if self.verbose:
                print(f'Loading previous model for iteration: {opts.model_load_iter} from {self.prev_model_path}')
            self.model.load_state_dict(torch.load(str(self.prev_model_path)))
        self.optimizer = self.create_optim(opts.optim, opts.lr)
        if self.verbose:
            print('Using optimizer:', f'[yellow]{opts.optim}(lr = {opts.lr})[/]', self.optimizer.state_dict())

        self.writer = SummaryWriter(str(self.output_dir))
        if self.verbose:
            print('Staring tensorboard session. Run following command to see the records.')
            print(f'$ [yellow]conda activate {self.opts.project_name} && tensorboard --logdir {self.output_dir}[/]')

    def model_path(self, iteration: int, check_exists: bool = False):
        model_name = f'model_{iteration:04}' if iteration > 0 else 'model_latest'
        model_path = self.output_dir / Path(model_name + '.pth')
        if check_exists and not model_path.exists():
            raise Exception(f'Cannot find model of iteration: {iteration}')
        return model_path

    @property
    def verbose(self):
        return self.opts.verbose

    def get_latest_iteration(self):
        return int(sorted(f.stem[7:] for f in self.output_dir.iterdir() if re.match('model_[0-9]{4}.pth', f.name))[-1])

    def _load_prev_model(self, iteration: int):
        if iteration == 0:
            return None
        if iteration >= -1:
            return self.model_path(iteration, check_exists=True)

    def save_iteration(self, iteration: int, write_data: Dict[str, Any]):
        # Write to SummaryWriter
        if self.verbose and iteration % 10 == 0:
            print(f'{iteration=} ===================', write_data)
        for k, v in write_data.items():
            self.writer.add_scalar(k, v, iteration)
        if iteration % self.opts.model_save_every == 0:
            if self.verbose:
                print(f'Saving current state to {self.model_path(iteration, False)}')
            torch.save(self.model.state_dict(), str(self.model_path(iteration, check_exists=False)))
        # Update `model_latest.pth`
        torch.save(self.model.state_dict(), str(self.model_path(-1, check_exists=False)))

    def update_optim_lr(self, lr: float):
        prev_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        if prev_lr != lr:
            if self.verbose:
                print(f'Updated LR of optimizer from {prev_lr} to {lr}')
            self.optimizer = self.create_optim(self.opts.optim, lr)

    def create_optim(self, optim_name: str, lr: float) -> torch.optim.Optimizer:
        return getattr(torch.optim, optim_name)(self.model.parameters(), lr=lr)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
