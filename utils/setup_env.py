import torch
from rich import print


def print_torch():
    print(f'Pytorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')


def check_cuda():
    if not torch.cuda.is_available():
        print('Cannot find CUDA. Following operations might fail unintentionally.')
        return False
    return True
