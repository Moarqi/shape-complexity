This is the official implementation for our paper [Shape complexity estimation using VAE](https://arxiv.org/abs/2304.02766)

Using this repository, sorting visualizations of shape images can be created using three different shape complexity estimation methods (which you can find in `complexity.py`).

## Setup

Using conda, you can set up the environment using the configuration file as below.
```
conda env create -f environment.yml
conda activate shape-complexity
```

## Usage

Run `python main.py --help` to get an overview of the available arguments:

```
usage: main.py [-h] [--train] [--epochs EPOCHS] [--input INPUT] [--mpeg7_path MPEG7_PATH] [--output OUTPUT] [--fill_ratio_norm] [--take TAKE] [--take_random]

options:
  -h, --help            show this help message and exit
  --train
  --epochs EPOCHS
  --input INPUT
  --mpeg7_path MPEG7_PATH
                        Specify path to root folder of MPEG7 dataset. If set, uses a custom dataset loader.
  --output OUTPUT
  --fill_ratio_norm
  --take TAKE           take X images from the input folder
  --take_random         select images randomly
```

For training the VAEs, you can either use your own dataset or use the [MPEG7 Dataset](https://dabi.temple.edu/external/shape/MPEG7/dataset.html) for which a specific data loader is implemented. Use it by providing the path to the root of the dataset using `--train --mpeg7_path <data root> --epochs 100`.

Pretrained model snapshots are saved and/or can be provided in the `trained` directory.

To use a pretrained model and visualize the sorting, you can specify the input folder using the `--input` argument and provide the root folder to a PyTorch `ImageFolder`.

