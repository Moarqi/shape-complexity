from typing import Any, Callable, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader


def plot_samples(
    masks: Tensor, ratings: List, rows=10, cols=10, labels: Optional[Tensor] = None
):
    dpi = 150
    total = rows * cols
    n_samples, _, y, x = masks.shape

    extent = (0, x - 1, 0, y - 1)
    if labels is not None:
        label_map = {
            v: i + 1 for i, v in enumerate({int(v): int(v) for v in labels}.keys())
        }
        max_label = len(label_map) + 1

    if total != n_samples:
        raise Exception("shape mismatch")

    fig = plt.figure(figsize=(32, 16), dpi=dpi)
    for idx in np.arange(n_samples):
        ax = fig.add_subplot(rows, cols, idx + 1, xticks=[], yticks=[])

        if labels is None:
            plt.imshow(
                masks[idx].permute(1, 2, 0), extent=extent, cmap="gray", vmin=0, vmax=1
            )
        else:
            mask = masks[idx][0] * (label_map[int(labels[idx].item())])  # type: ignore
            plt.imshow(
                mask,
                cmap="turbo",
                extent=extent,
                vmin=0,
                vmax=max_label,  # type: ignore
            )

        rating = ratings[idx]
        ax.set_title(
            rating if isinstance(rating, str) else f"{ratings[idx]:.3f}",
            fontdict={
                "fontsize": 6 if y < 128 else 18,
                "color": "orange" if labels is None else "white",
            },
            y=0.2 if isinstance(rating, str) else 0.75,
        )

    fig.patch.set_facecolor("#292929") # type: ignore
    height_px = y * rows
    width_px = x * cols
    fig.set_size_inches(width_px / (dpi / 2), height_px / (dpi / 2), forward=True)
    fig.tight_layout(pad=0)

    return fig


def visualize_sort(
    data_loader: DataLoader,
    metric_fn: Callable[[Any], tuple[float, Optional[Tensor]]],
    metric_name: str,
    *fn_args: Any,
    plot_ratings=False,
    rows=10,
    cols=10,
    **fn_kwargs: Any,
):
    n_samples = len(data_loader.sampler)  # type: ignore
    recon_masks: Optional[torch.Tensor] = None
    masks = torch.zeros((n_samples, 1, 64, 64))
    ratings = torch.zeros((n_samples,))

    plot_recons = True

    for i, (mask, _) in enumerate(data_loader, 0):
        rating, mask_recon_grid = metric_fn(mask, *fn_args, **fn_kwargs)
        if plot_recons and mask_recon_grid == None:
            plot_recons = False
        elif plot_recons and recon_masks is None and mask_recon_grid is not None:
            recon_masks = torch.zeros((n_samples, *mask_recon_grid.shape))

        masks[i] = mask[0]
        ratings[i] = rating

        if plot_recons:
            recon_masks[i] = mask_recon_grid  # type: ignore

    if plot_ratings:
        plt.plot(np.arange(len(ratings)), np.sort(ratings.numpy()))
        plt.xlabel("images")
        plt.ylabel(f"{metric_name} rating")
        plt.savefig(f"results/{metric_name}_rating_plot.pdf")
        plt.close()

    sort_idx = torch.argsort(ratings)

    masks_sorted = masks[sort_idx]
    fig = plot_samples(masks_sorted, ratings[sort_idx].numpy(), rows=rows, cols=cols)
    fig.savefig(f"results/{metric_name}_sort.pdf")
    plt.close(fig)

    if plot_recons and recon_masks is not None:
        recon_masks_sorted = recon_masks[sort_idx]
        fig_recon = plot_samples(
            recon_masks_sorted, ratings[sort_idx].numpy(), rows=rows, cols=cols
        )
        fig_recon.savefig(f"results/{metric_name}_sort_recon.pdf")
        plt.close(fig_recon)

    return sort_idx, ratings


def visualize_fixed_sort(
    data_loader: DataLoader,
    sort_idx: torch.Tensor,
    metric_name: str,
    rows=10,
    cols=10,
):
    n_samples = len(data_loader.sampler)  # type: ignore
    assert n_samples == len(sort_idx), "sort idx length mismatch"

    masks = torch.zeros((n_samples, 1, 64, 64))

    for i, (mask, _) in enumerate(data_loader, 0):
        masks[i] = mask[0]

    ratings = sort_idx / sort_idx.max()

    masks_sorted = masks[sort_idx]
    fig = plot_samples(masks_sorted, ratings[sort_idx].numpy(), rows=rows, cols=cols)
    fig.savefig(f"results/{metric_name}_sort.pdf")
    plt.close(fig)

    return sort_idx, None


def visualize_sort_multidim(
    data_loader: DataLoader,
    measures: list[tuple[str, Callable[[Any], tuple[float, Tensor]], Any]],
    rows=10,
    cols=10,
    max_norm=False,
    use_labels=False,
):
    n_samples = len(data_loader.sampler)  # type: ignore
    n_dim = len(measures)
    masks = torch.zeros((n_samples, 1, 64, 64))
    ratings = torch.zeros((n_samples, n_dim))
    labels = torch.zeros((n_samples,))

    for i, (mask, label) in enumerate(data_loader, 0):
        masks[i] = mask[0]
        labels[i] = label

        for m in range(n_dim):
            _, fn, args = measures[m]
            rating, _ = fn(mask, *args)
            ratings[i, m] = rating

    if max_norm:
        mins = ratings.min(dim=0)
        maxs = ratings.max(dim=0)
        ratings[:] -= mins.values
        ratings[:] /= maxs.values - mins.values

    # if n_dim == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection="3d")
    #     ax.scatter(ratings[:, 0], ratings[:, 1], ratings[:, 2], marker="o")

    #     ax.set_xlabel(measures[0][0])
    #     ax.set_ylabel(measures[1][0])
    #     ax.set_zlabel(measures[2][0])
    #     plt.savefig(f"results/3d_plot{'_norm' if max_norm else ''}.pdf")
    #     plt.close()

    measure_norm = torch.linalg.vector_norm(ratings, dim=1)
    sort_idx = torch.argsort(measure_norm)
    rating_strings = [f"{r[0]:.3f}\n{r[1]:.3f}\n{r[2]:.3f}" for r in ratings[sort_idx]]

    fig = plot_samples(
        masks[sort_idx],
        rating_strings,
        labels=labels[sort_idx] if use_labels else None,
        rows=rows,
        cols=cols,
    )
    fig.savefig(
        f"results/{n_dim}dim_{'_'.join([m[0] for m in measures])}_sort{'_norm' if max_norm else ''}.pdf"
    )
    plt.close(fig)

    return sort_idx, measure_norm
