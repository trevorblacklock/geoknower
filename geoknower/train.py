from collections.abc import Callable
from pathlib import Path

import healpy as hp
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .grid import colatlon_to_cart, latlon_to_cart, pix_to_cart
from .model import GeoModel

__all__ = [
    "exact_latlon",
    "gaussian_cross_entropy",
    "gaussian_mask",
    "gaussian_targets",
    "is_within_distance",
    "train_model",
]


def gaussian_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss function for outputs of network
    """
    logp = nn.functional.log_softmax(logits, dim=1)
    return -torch.mean(torch.sum(target * logp, dim=1))


def is_within_distance(
    logits: torch.Tensor,
    locs: torch.Tensor,
    centers: torch.Tensor,
    km: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given the logits from the model, centers of each corresponding pixel,
    and unit vectors of each expected position, returns True if corresponding
    model prediction was within the specified distance.
    """
    r_earth = 6371

    logits = logits.squeeze(0) if logits.ndim == 1 else logits
    locs = logits.squeeze(0) if locs.ndim == 1 else locs
    centers = centers.squeeze(0) if centers.ndim == 1 else centers

    probs = torch.softmax(logits, dim=1)

    if mask is not None:
        probs = probs * mask[torch.argmax(probs, dim=1)]
        probs = probs / torch.clamp_min(torch.sum(probs, dim=1, keepdim=True), 1e-12)

    cart = probs @ centers
    cart = cart / torch.clamp_min(torch.norm(cart, dim=1, keepdim=True), 1e-12)

    cross = torch.norm(torch.cross(cart, locs, dim=1), dim=1)
    dot = torch.clamp(torch.sum(cart * locs, dim=1), min=-1, max=1)
    ang = torch.atan2(cross, dot)

    return ang * r_earth <= km


def exact_latlon(
    logits: torch.Tensor, centers: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given the logits from the model, centers of each corresponding pixel,
    returns the exact weighted lat/lon predicted by the model.
    """
    logits = logits.squeeze(0) if logits.ndim == 1 else logits
    centers = centers.squeeze(0) if centers.ndim == 1 else centers

    probs = torch.softmax(logits, dim=1)

    if mask is not None:
        probs = probs * mask[torch.argmax(probs, dim=1)]
        probs = probs / torch.clamp_min(torch.sum(probs, dim=1, keepdim=True), 1e-12)

    cart = probs @ centers
    cart = cart / torch.clamp_min(torch.norm(cart, dim=1, keepdim=True), 1e-12)

    lat = torch.rad2deg(torch.asin(-cart[:, 2]))
    lon = torch.rad2deg(torch.atan2(-cart[:, 1], -cart[:, 0]))

    return lat, lon


def gaussian_targets(
    lat: ArrayLike, lon: ArrayLike, nside: int, pixgrid: ArrayLike, alpha: float = 10.0
) -> torch.Tensor:
    """
    Gaussian targets for an expected input location.
    """
    theta, phi = hp.pix2ang(nside, pixgrid)
    centers = colatlon_to_cart(theta, phi)
    locs = latlon_to_cart(lat, lon)

    centers = np.reshape(centers, shape=(-1, 3))
    locs = np.reshape(locs, shape=(-1, 3))

    sigma = 1 - np.cos(alpha * hp.max_pixrad(nside))
    mask = 1 - locs @ centers.T
    sigma = torch.as_tensor(sigma)
    mask = torch.as_tensor(mask)

    return torch.softmax(-mask / sigma**2, dim=1, dtype=torch.float32)


def gaussian_mask(nside: int, pixgrid: ArrayLike, alpha: float = 10.0) -> torch.Tensor:
    """
    Gaussian mask for a input healpy pixel grid.
    """
    centers = pix_to_cart(nside, pixgrid)
    sigma = 1 - np.cos(alpha * hp.max_pixrad(nside))
    mask = 1 - centers @ centers.T

    sigma = torch.as_tensor(sigma)
    mask = torch.as_tensor(mask)

    return torch.softmax(-mask / sigma**2, dim=1, dtype=torch.float32)


def train_model(
    model: GeoModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    opt: Optimizer,
    train_targets: ArrayLike,
    val_targets: ArrayLike,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    val_fn: Callable[..., torch.Tensor],
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "geomodel",
    checkpoint_rate: int = 0,
    autocast: bool = True,
    gradscaler: bool = True,
    device: str = "cuda",
    **kwargs,
):
    if checkpoint_rate != 0:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_targets = torch.as_tensor(train_targets, dtype=torch.float32)
    val_targets = torch.as_tensor(val_targets, dtype=torch.float32)
    scaler = torch.GradScaler(device, enabled=gradscaler)

    train_batch_size = (
        1 if train_dataloader.batch_size is None else train_dataloader.batch_size
    )
    val_batch_size = 1 if val_dataloader.batch_size is None else val_dataloader.batch_size
    num_train = len(train_dataloader) * train_batch_size
    num_val = len(val_dataloader) * val_batch_size

    checkpoint_num = 1
    model = model.to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for idx, images in train_dataloader:
            images = images.to(device, non_blocking=True)
            target = train_targets[idx].to(device, non_blocking=True)

            with torch.autocast(device, enabled=autocast):
                logits = model(images)
                loss = loss_fn(logits, target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * idx.shape[0]

        model.eval()
        accuracy = 0.0

        for idx, images in val_dataloader:
            images = images.to(device, non_blocking=True)
            target = val_targets[idx].to(device, non_blocking=True)

            with torch.no_grad():
                logits = model(images)
                predicate = val_fn(logits, target, **kwargs)
                accuracy += torch.sum(predicate)

        print(
            f"Epoch: {epoch + 1} | Loss: {running / num_train:.6f}"
            f" | Accuracy: {100 * accuracy / num_val:.6f}%"
        )

        if checkpoint_rate != 0 and (epoch + 1) % checkpoint_rate == 0:
            fdir = Path(checkpoint_dir, f"{checkpoint_name}-ckpt-{checkpoint_num}.pt")
            print(f"Saving a checkpoint for epoch {epoch + 1} under {fdir}.")
            torch.save(model.state_dict(), fdir)
            checkpoint_num += 1
