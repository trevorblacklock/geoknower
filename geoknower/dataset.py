import asyncio
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from numpy.typing import ArrayLike
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from .streetview import (
    batch_async_pano,
    batch_async_pano_metadata,
    batch_async_streetview_metadata,
    get_streetview_session,
)

__all__ = [
    "PanoDataset",
    "build_dataset",
    "bytes_to_image",
    "image_to_bytes",
    "load_credentials",
]


class PanoDataset(Dataset):
    """
    Given a HF dataset and a compositor creates a dataset compatible with PyTorch's
    DataLoader. Will work on all images in the HF dataset, so be careful to only include
    relevant image columns.
    """

    def __init__(self, data, compositor: Compose):
        if not isinstance(data, HFDataset):
            raise TypeError("Passed data must be a HFDataset.")

        self.data = data
        self.compositor = compositor

        image_cols = [
            name
            for name, feature in self.data.features.items()
            if isinstance(feature, HFImage)
        ]
        self.images = self.data.select_columns(image_cols)

    def __len__(self) -> int:
        return self.data.num_rows

    def __getitem__(self, index) -> tuple[int, torch.Tensor]:
        processed = [self.compositor(image) for image in self.images[index].values()]
        return (index, torch.stack(processed))


def load_credentials(path: str) -> dict[str, str]:
    """
    Load API keys from json file
    """
    fpath = Path(path)
    with open(fpath) as file:
        return json.load(file)


def image_to_bytes(image: Image.Image) -> bytes:
    """
    Convert a PIL image to bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_image(bytes: bytes) -> Image.Image:
    """
    Convert bytes to a PIL image
    """
    return Image.open(io.BytesIO(bytes))


def build_dataset(
    lat: ArrayLike,
    lon: ArrayLike,
    zoom: int,
    api_key: str,
    signing_key: str,
    session_key: str | None = None,
    radius: ArrayLike = 100,
    concurrency: int = 1,
    verbose: bool = True,
    raise_on_missing: bool = True,
) -> pd.DataFrame:
    """
    Build a dataframe from known lat/lon positions. Panos are only taken from
    official Google coverage, and will raise error if missing any metadata.
    """
    lat, lon = np.asarray((lat, lon))
    radius = np.resize(radius, new_shape=lat.shape)

    sv_metadata = asyncio.run(
        batch_async_streetview_metadata(
            lat,
            lon,
            radius,
            api_key,
            signing_key,
            concurrency=concurrency,
            verbose=verbose,
        )
    )

    sv_mask = [
        idx
        for idx, data in enumerate(sv_metadata)
        if data and data["status"] == "OK" and "Google" in data["copyright"]
    ]
    covered = [sv_metadata[idx] for idx in sv_mask]

    lat = lat[sv_mask]
    lon = lon[sv_mask]

    if raise_on_missing and len(covered) != len(sv_metadata):
        raise RuntimeError(
            "Could not find valid streetview metadata for all given locations."
        )

    if session_key is None:
        session_data = get_streetview_session(api_key)
        session_key = session_data["session"]

    pano_ids = [data["pano_id"] for data in covered if data]
    dates = [data["date"] for data in covered if data]
    pano_metadata = asyncio.run(
        batch_async_pano_metadata(
            pano_ids,
            api_key,
            session_key,
            concurrency=concurrency,
            verbose=verbose,
        )
    )

    pano_mask = [
        idx for idx, data in enumerate(pano_metadata) if data and "panoId" in data
    ]
    verified_pano_metadata = [pano_metadata[idx] for idx in pano_mask]

    lat = lat[pano_mask]
    lon = lon[pano_mask]

    if raise_on_missing and len(verified_pano_metadata) != len(sv_metadata):
        raise RuntimeError(
            "Could not find valid panorama metadata for all given locations."
        )

    nx = 2**zoom
    ny = 2 ** (zoom - 1) if zoom >= 1 else 1
    coords = list(itertools.product(range(nx), range(ny)))

    raw_panos = asyncio.run(
        batch_async_pano(
            pano_ids, zoom, api_key, session_key, concurrency=concurrency, verbose=verbose
        )
    )

    panos_bytes = [
        [bytes(0) if tile is None else tile for tile in tiles] for tiles in raw_panos
    ]

    metadata_dict = {"pano_id": pano_ids, "lat": lat, "lon": lon, "date": dates}
    image_dict = {
        f"img{x}_{y}": data for (x, y), data in zip(coords, list(zip(*panos_bytes)))
    }
    combined_dict = metadata_dict | image_dict

    return pd.DataFrame(combined_dict)
