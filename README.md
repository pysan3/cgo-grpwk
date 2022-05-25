# cgo-grpwk

## Documents

- [Initial Report (Not available for the public)](https://hackmd.io/XXXX)

## Start Up

### Create Environment

```bash
$ git clone git@github.com:pysan3/cgo-grpwk.git
$ cd cgo-grpwk
$ conda env create -f ./environment.yml
$ conda activate cgo-grpwk

# install pip dependencies
$ pip install torchtyping
```

## Run Script

```bash
# conda activate cgo-grpwk
$ python main.py
```

### Useful Options

```bash
$ python main.py --flagfile params.txt  # tweak default options
# options currently in this file are the defaults

$ python main.py --noverbose  # suppress outputs

$ python main.py --model_out out_$(date "+%Y-%m-%d_%H-%M")  # save training outputs to a time dependent folder
# dirs starting with `./out`... will be git ignored
```

For more details, look at section [CodeHelp](#codehelp)

## Result

- Training Loss etc can be viewed with `tensorboard`
  - For more details, read instructions shown right after executing the code.
  - `$ tensorboard --logdir <model_out>`
- Rendered images during training are stored in `<model_out>/images/<iter>.png`
- Trained model parameters are stored in `<model_out>/model_latest.pth`
  - Should be loaded with `torch.load(<file_name>)`
- **Final rendered image is created at `<model_out>/eval.png`**

## Additional Info

### CodeHelp

```txt
       USAGE: main.py [flags]
flags:
  --batch_size: Size of batch while training
    (default: '10')
    (an integer)
  --lr: learning rate
    (default: '0.1')
    (a number)
  --model_load_iter: Load model at this iteration. (-1) means the latest
    (default: '0')
    (an integer)
  --model_out: Path to output model and training progress
    (default: './out')
  --model_save_every: Interval of saving current model weights
    (default: '10')
    (an integer)
  --num_iters: How many iterations for training
    (default: '300')
    (an integer)
  --optim: Name of optimizer. Will load as `getattr(torch.optim, opts.optim)`
    (default: 'Adam')
  --project_name: Name of this project. Do not change this from default value.
    (default: 'cgo-grpwk')
  --[no]verbose: Print more detailed informations during training.
    (default: 'true')
  --[no]vis: Toggle to visualize the rendered result during training
    (default: 'false')
  --vol_extent_world: Our rendered scene is centered around (0,0,0) and is
    enclosed inside a bounding box
    (default: '3.0')
    (a number)
  --vol_size: Size of volume to express the space
    (default: '128')
    (an integer)

Try --helpfull to get a list of all flags.
```

## License

All files in this repository are licensed under the MIT license except files under [./tutorials](./tutorials) folder.
The license for our files are detailed in [LICENSE](./LICENSE).
Files in [./tutorials](./tutorials) are taken from [pytorch3d](https://github.com/facebookresearch/pytorch3d) repository and they follow the BSD License as described [here](https://github.com/facebookresearch/pytorch3d#license).
