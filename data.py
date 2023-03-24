import os
from typing import Any, Callable, Optional
import numpy as np
import torch

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder


class MPEG7ShapeDataset(Dataset):
    """helper class for interacting with the MPEG7 Dataset"""

    img_dir: str
    filenames: list[str] = []
    labels: list[int] = []
    label_dict: dict[int, str]
    transform: Optional[Callable] = None

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        paths = os.listdir(self.img_dir)
        labels = []
        for file in paths:
            fp = os.path.join(self.img_dir, file)
            if os.path.isfile(fp):
                label = file.split("-")[0].lower()
                self.filenames.append(fp)
                labels.append(label)

        label_name_dict = dict.fromkeys(labels)

        self.label_dict = {i: v for (i, v) in enumerate(label_name_dict.keys())}
        self.label_index_dict = {v: i for (i, v) in self.label_dict.items()}
        self.labels = [self.label_index_dict[l] for l in labels]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        image.convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(img):
    img = img.numpy()
    max_x, max_y = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = rmin - 1 if rmin > 0 else rmin
    cmin = cmin - 1 if cmin > 0 else cmin
    rmax = rmax + 1 if rmax < max_x else rmax
    cmax = cmax + 1 if cmax < max_y else cmax

    return rmin, rmax, cmin, cmax


class BBoxTransform:
    squared: bool = False

    def __init__(self, squared: bool = False) -> None:
        if squared is not None:
            self.squared = squared

    def __call__(self, img: Any) -> Tensor:
        img = transforms.F.to_tensor(img).squeeze(dim=0)
        ymin, ymax, xmin, xmax = bbox(img)
        if not self.squared:
            return transforms.F.to_pil_image(img[ymin:ymax, xmin:xmax].unsqueeze(dim=0))

        max_dim = (ymin, ymax) if ymax - ymin > xmax - xmin else (xmin, xmax)
        n = max_dim[1] - max_dim[0]
        if n % 2 != 0:
            n += 1

        n_med = np.round(n / 2)

        ymedian = np.round(ymin + (ymax - ymin) / 2)
        xmedian = np.round(xmin + (xmax - xmin) / 2)

        M, N = img.shape

        ycutmin, ycutmax = (
            int(ymedian - n_med if ymedian >= n_med else 0),
            int(ymedian + n_med if ymedian + n_med <= M else M),
        )

        xcutmin, xcutmax = (
            int(xmedian - n_med if xmedian >= n_med else 0),
            int(
                xmedian + n_med if xmedian + n_med <= N else N,
            ),
        )

        if (ycutmax - ycutmin) % 2 != 0:
            ycutmin += 1
        if (xcutmax - xcutmin) % 2 != 0:
            xcutmin += 1

        squared_x = np.zeros((n, n))
        squared_cut_y = np.round((ycutmax - ycutmin) / 2)
        squared_cut_x = np.round((xcutmax - xcutmin) / 2)

        dest_ymin, dest_ymax = int(n_med - squared_cut_y), int(n_med + squared_cut_y)
        dest_xmin, dest_xmax = int(n_med - squared_cut_x), int(n_med + squared_cut_x)

        squared_x[
            dest_ymin:dest_ymax,
            dest_xmin:dest_xmax,
        ] = img[ycutmin:ycutmax, xcutmin:xcutmax]

        return transforms.F.to_pil_image(torch.from_numpy(squared_x).unsqueeze(dim=0))


def get_image_transform(is_train=False):
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomRotation(degrees=85, expand=True),
            BBoxTransform(squared=True),
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
        if is_train
        else [
            transforms.Grayscale(),
            BBoxTransform(squared=True),
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ]
    )


def load_mpeg7_data(path: str, is_train=False):
    transform = get_image_transform(is_train=is_train)

    return MPEG7ShapeDataset(path, transform)


def load_input_data(path: str, is_train=False):
    transform = get_image_transform(is_train=is_train)

    return ImageFolder(path, transform)
