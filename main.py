import argparse
import os
import numpy as np

import torch
from complexity import compression_measure, fft_measure, vae_reconstruction_measure

from models import load_models
from data import get_image_transform, load_input_data, load_mpeg7_data
from torch.utils.data import DataLoader, Subset

from plot import visualize_sort, visualize_sort_multidim


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train", action="store_true", help="Train the VAEs")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training"
    )
    parser.add_argument(
        "--input", type=str, default="images", help="Path to input ImageFolder"
    )
    parser.add_argument(
        "--mpeg7_path",
        type=str,
        default=None,
        help="Specify path to root folder of MPEG7 dataset. If set, uses a custom dataset loader.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to visualization output folder",
    )
    parser.add_argument(
        "--fill_ratio_norm",
        action="store_true",
        help="Normalize the measures by the fill ratio of the images",
    )
    parser.add_argument(
        "--take", type=int, default=20, help="take X images from the input folder"
    )
    parser.add_argument(
        "--take_random", action="store_true", help="select images randomly"
    )

    return parser


def main():
    args = get_argparser().parse_args()
    model_bn16, model_bn64 = load_models(load_pretrained=not args.train, beta=1)
    data = (
        load_input_data(args.input, args.train)
        if args.mpeg7_path is None
        else load_mpeg7_data(args.mpeg7_path, args.train)
    )
    data_loader = DataLoader(data, batch_size=256 if args.train else 1)

    if args.train:
        if not os.path.exists("trained"):
            os.makedirs("trained")
        min_loss64 = np.inf
        min_loss16 = np.inf

        for epoch in range(args.epochs):
            avg_loss_bn64 = model_bn64.train_loop(epoch, data_loader)
            avg_loss_bn16 = model_bn16.train_loop(epoch, data_loader)

            if avg_loss_bn64 < min_loss64:
                torch.save(
                    model_bn64.state_dict(),
                    f"trained/{model_bn64.__class__.__name__}{model_bn64.bottleneck}_beta{model_bn64.beta}.pth",
                )

            if avg_loss_bn16 < min_loss16:
                torch.save(
                    model_bn16.state_dict(),
                    f"trained/{model_bn16.__class__.__name__}{model_bn16.bottleneck}_beta{model_bn16.beta}.pth",
                )

    model_bn64.eval()
    model_bn16.eval()

    data.transform = get_image_transform(is_train=False)
    data_len = len(data)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if data_len > args.take:
        if args.take_random:
            indices = torch.randperm(data_len)[: args.take]
            subset = Subset(data, indices.tolist())
            data_loader = DataLoader(subset, batch_size=1)
        else:
            data_loader = DataLoader(
                data, batch_size=1, sampler=list(range(args.take))
            )  # type: ignore

        data_len = args.take

    visualize_sort(
        data_loader,
        vae_reconstruction_measure,  # type: ignore
        "vae",
        model_bn64,
        model_bn16,
        rows=1,
        cols=data_len,
        fill_ratio_norm=args.fill_ratio_norm,
    )

    visualize_sort(
        data_loader,
        compression_measure,
        "compression",
        rows=1,
        cols=data_len,
        fill_ratio_norm=args.fill_ratio_norm,
    )

    visualize_sort(
        data_loader,
        fft_measure,
        "fft",
        rows=1,
        cols=data_len,
    )

    visualize_sort_multidim(
        data_loader,
        [
            ("fft", fft_measure, []),
            ("compression", compression_measure, [True]),
            (
                "vae",
                vae_reconstruction_measure,
                [model_bn64, model_bn16, False],
            ),
        ],  # type: ignore
        max_norm=True,
        rows=1,
        cols=data_len,
    )


if __name__ == "__main__":
    main()
