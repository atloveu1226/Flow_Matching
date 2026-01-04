import os

import jax
from absl import logging
import imageio
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import matplotlib as mpl

mpl.rcParams["savefig.pad_inches"] = 0
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def reshape2image(data, config: DictConfig):
    """
    Normalize arbitrary shaped image data to (N, C, H, W).

    Accepts flattened arrays with shape (d, N) or (N, d), batched image tensors
    shaped like (N, C, H, W) or (N, H, W, C), and single-image inputs such as
    (C, H, W) or (H, W, C). The spatial dimensions come from config.data.shape.
    """
    data = np.asarray(data)
    logging.info(f"Reshaping data with original shape {data.shape}, sum {data.sum()}")

    data_shape = config.data.shape  # (H, W, C)
    if len(data_shape) != 3:
        raise ValueError(f"config.data.shape must be length 3, got {data_shape}")

    h, w, c = data_shape
    flat_dim = h * w * c

    if data.ndim == 1:
        data = data[:, None]

    if data.ndim == 2:
        if data.shape[0] == flat_dim and data.shape[1] != flat_dim:
            data = np.ascontiguousarray(data.T)
        elif data.shape[1] == flat_dim:
            data = np.ascontiguousarray(data)
        else:
            raise ValueError(
                f"Expected data with flattened dimension {flat_dim}, got {data.shape}"
            )
        data = data.reshape(-1, c, h, w)
    elif data.ndim == 3:
        if data.shape == (c, h, w):
            data = data[None, ...]
        elif data.shape == (h, w, c):
            data = np.transpose(data, (2, 0, 1))[None, ...]
        elif c == 1 and data.shape[1:] == (h, w):
            data = data[:, None, :, :]
        else:
            raise ValueError(
                f"Unable to reshape data of shape {data.shape} into (N, {c}, {h}, {w})"
            )
    elif data.ndim == 4:
        if data.shape[1:] == (c, h, w):
            pass
        elif data.shape[1:] == (h, w, c):
            data = np.transpose(data, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"Unable to reshape data of shape {data.shape} into (N, {c}, {h}, {w})"
            )
    else:
        raise ValueError(
            "data must be reshaped to (d, N), (N, d), (N, C, H, W), or (N, H, W, C) before calling reshape2image"
        )

    return data


def save_image(
    config: DictConfig, i, data, nrow=None, prefix="samples", ground_truth=None
):
    """
    Saves image and returns the saved image path.
    """
    img_path = _save_image(config, i, data, nrow=nrow, prefix=prefix)
    return img_path


def _save_image(config: DictConfig, i, data, nrow=None, prefix="samples"):
    """
    Save images in a grid format using PIL.

    Args:
        config: DictConfig, configuration object
        i: int, current iteration step for naming the file
        data: (d,N) array-like, data to be saved as images
        nrow: int, number of images per row in the grid
        prefix: str, prefix for the saved image filename
    """
    data = np.asarray(data)
    data = (data + 1.0) / 2.0
    data = reshape2image(data, config)

    h = w = config.data.image_size
    c = config.data.num_channels

    N = data.shape[0]

    if nrow is None:
        nrow = int(np.sqrt(max(N, 1)))
        nrow = max(1, nrow)
    ncol = nrow
    rows = int(np.ceil(N / ncol))
    data = np.clip(data, 0.0, 1.0)
    data_uint8 = (data * 255).astype(np.uint8)

    # create grid
    mode = "L" if c == 1 else "RGB"
    grid_height = rows * h
    grid_width = ncol * w
    channels = 1 if mode == "L" else 3
    grid = np.zeros((grid_height, grid_width, channels), dtype=np.uint8)

    for idx in range(N):
        r = idx // ncol
        c = idx % ncol
        if mode == "L":
            tile = data_uint8[idx, 0]  # h*w
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w, 0] = tile
        else:
            # c*h*w -> h*w*c
            tile = np.transpose(data_uint8[idx], (1, 2, 0))
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w, :] = tile

    images_dir = os.path.join(config.logging.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(images_dir, f"{prefix}_{i:04d}.png")
    img = Image.fromarray(grid.squeeze() if mode == "L" else grid, mode=mode)
    img.save(out_path)
    return out_path
