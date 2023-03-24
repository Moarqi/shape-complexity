from bz2 import compress
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torchvision.utils import make_grid


FFT_MEASURE_MAX = np.sqrt(np.power(0.5, 2) + np.power(0.5, 2))


def fill_ratio(img: Tensor):
    np_img = img[0].numpy()
    fill_ratio = np_img.sum().item() / np.ones_like(np_img).sum().item()

    return 1 - fill_ratio, None


def compression_measure(
    img: Tensor, fill_ratio_norm=False
) -> tuple[float, Optional[Tensor]]:
    np_img = img[0].numpy()
    np_img_bytes = np_img.tobytes()
    compressed = compress(np_img_bytes)

    complexity = len(compressed) / len(np_img_bytes)

    if fill_ratio_norm:
        fill_ratio = np_img.sum().item() / np.ones_like(np_img).sum().item()
        return complexity * (1 - fill_ratio), None

    return complexity, None


def fft_measure(img: Tensor):
    np_img = img[0][0].numpy()
    fft = np.fft.fft2(np_img)

    fft_abs = np.abs(fft)

    n = fft.shape[0]
    pos_f_idx = n // 2
    df = np.fft.fftfreq(n=n)  # type: ignore

    amplitude_sum = fft_abs[:pos_f_idx, :pos_f_idx].sum()
    mean_x_freq = (fft_abs * df)[:pos_f_idx, :pos_f_idx].sum() / amplitude_sum
    mean_y_freq = (fft_abs.T * df).T[:pos_f_idx, :pos_f_idx].sum() / amplitude_sum

    mean_freq = np.sqrt(np.power(mean_x_freq, 2) + np.power(mean_y_freq, 2))

    # mean frequency in range 0 to np.sqrt(0.5^2 + 0.5^2)
    return mean_freq / FFT_MEASURE_MAX, None


def vae_reconstruction_measure(
    img: Tensor,
    model_gb: nn.Module,
    model_lb: nn.Module,
    fill_ratio_norm=False,
) -> tuple[float, Optional[Tensor]]:
    model_gb.eval()
    model_lb.eval()

    with torch.no_grad():
        mask = img.to(model_gb.device)  # type: ignore

        recon_gb: Tensor
        recon_lb: Tensor

        recon_gb, _, _ = model_gb(mask)
        recon_lb, _, _ = model_lb(mask)

        abs_px_diff = (recon_gb - recon_lb).abs().sum().item()

        complexity = abs_px_diff / mask.sum()

        if fill_ratio_norm:
            complexity *= mask.sum().item() / torch.ones_like(mask).sum().item()

        return (
            complexity,
            make_grid(
                torch.stack(
                    [mask[0], recon_gb.view(-1, 64, 64), recon_lb.view(-1, 64, 64)]
                ).cpu(),
                nrow=1,
                padding=0,
            ),
        )


def multidim_complexity(
    img: Tensor,
    measures: list[tuple[str, Callable[[Any], tuple[float, Tensor]], Any]],
):
    n_dim = len(measures)
    ratings = torch.zeros((n_dim,))

    for i, (_, fn, args) in enumerate(measures):
        rating, _ = fn(img, *args)
        ratings[i] = rating

    return ratings
