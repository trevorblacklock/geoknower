from pathlib import Path

import healpy as hp
import numpy as np
from numpy.typing import ArrayLike, NDArray

type LatLon = tuple[float, float]
type LatLons = tuple[NDArray, NDArray]

__all__ = [
    "colatlon_to_cart",
    "colatlon_to_latlon",
    "generate_uniform_latlon",
    "latlon_to_cart",
    "latlon_to_colatlon",
    "latlon_to_pix",
    "nside",
    "pix_to_cart",
    "pix_to_latlon",
    "pixgrid",
    "pixgrid_to_poly",
]


MAP_LOCAL_PATH = "data/cmb-pixgrid.npy"
MAP_PATH = Path(__file__).resolve().parent / MAP_LOCAL_PATH

pixgrid_res: int = 6
nside: int = 2**pixgrid_res
pixgrid: NDArray = np.load(MAP_PATH)


def colatlon_to_latlon(lat: ArrayLike, lon: ArrayLike) -> LatLons:
    """
    Given a co-(lats/lons), convert them to lats/lons.
    """
    return np.degrees(lat) - 90, np.degrees(lon) - 180


def latlon_to_colatlon(lat: ArrayLike, lon: ArrayLike) -> LatLons:
    """
    Given a lats/lons, convert them to co-(lats/lons).
    """
    return np.radians(lat) + np.pi / 2, np.radians(lon) + np.pi


def pix_to_latlon(nside: int, pix: ArrayLike) -> LatLons:
    """
    Convert a set of healpix pixels to lat/lon.
    """
    lat, lon = hp.pix2ang(nside, pix)
    return colatlon_to_latlon(lat, lon)


def latlon_to_pix(nside: int, lat: ArrayLike, lon: ArrayLike) -> NDArray:
    """
    Convert a set of lat/lon to healpix pixels.
    """
    lat, lon = latlon_to_colatlon(lat, lon)
    return hp.ang2pix(nside, lat, lon)


def colatlon_to_cart(lat: ArrayLike, lon: ArrayLike) -> NDArray:
    """
    Convert a set of co-(lat/lon) to normalized cartesian vectors.
    """
    return np.stack(
        [
            np.sin(lat) * np.cos(lon),
            np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ],
        axis=-1,
    )


def latlon_to_cart(lat: ArrayLike, lon: ArrayLike) -> NDArray:
    """
    Convert a set of lat/lon to normalized cartesian vectors.
    """
    lat, lon = latlon_to_colatlon(lat, lon)
    return colatlon_to_cart(lat, lon)


def pix_to_cart(nside: int, pix: ArrayLike) -> NDArray:
    """
    Convert a set of healpix pixels to normalized cartesian vectors.
    """
    theta, phi = hp.pix2ang(nside, pix)
    return colatlon_to_cart(theta, phi)


def generate_uniform_latlon(
    nside: int,
    attempts: int,
    map_data: ArrayLike,
    lat_range: LatLon = (0, np.pi),
    lon_range: LatLon = (0, 2 * np.pi),
    colatlon: bool = False,
) -> LatLons:
    """
    Calculate a uniform number of lats/lons contained within the given map. Number of
    attempts will not necessarily equal the number of output points.
    """
    map_data = np.asarray(map_data)
    generator = np.random.default_rng()
    lats, lons = (
        generator.uniform(*lat_range, size=attempts),
        generator.uniform(*lon_range, size=attempts),
    )

    pix = hp.ang2pix(nside, lats, lons)
    mask = np.isin(pix, map_data)

    if colatlon:
        return lats[mask], lons[mask]
    return colatlon_to_latlon(lats[mask], lons[mask])


def _pix_boundaries(nside: int, pix: ArrayLike, step: int = 1) -> LatLons:
    nvert = 4 * step
    bvecs = hp.boundaries(nside, pix, step=step).mT
    blat, blon = np.asarray(hp.vec2ang(bvecs))

    blat, blon = colatlon_to_latlon(blat, blon)
    blat = np.reshape(blat, shape=(-1, nvert))
    blon = np.reshape(blon, shape=(-1, nvert))

    return blat, blon


def _candidate_boundary_indices(lon: ArrayLike) -> NDArray:
    lon = np.asarray(lon)
    diff = np.abs(np.max(lon, axis=-1) - np.min(lon, axis=-1))
    return np.squeeze(np.where(diff > 180))


def _transform_candidate(lon: ArrayLike, eps: float = 1e-6) -> NDArray:
    lon = np.asarray(lon)
    return lon[lon < -eps] + 360


def _accepted_candidate_indices(lon: ArrayLike, eps: float = 1e-6) -> NDArray:
    lon = np.asarray(lon)
    return np.squeeze(np.where(np.max(lon, axis=-1) > 180 + eps))


def _split_polygon(
    lat: ArrayLike, lon: ArrayLike, bisection: float = 180, eps: float = 1e-6
) -> tuple[NDArray, NDArray]:
    polygon = np.stack((lat, lon), axis=-1)
    left = polygon[polygon[..., 1] < bisection - eps]
    right = polygon[polygon[..., 1] > bisection + eps]
    intersections = polygon[np.isclose(polygon[..., 1], 180, atol=eps, rtol=eps)]

    if intersections.shape[0] == 2:
        i1, i2 = intersections
    elif intersections.shape[0] == 1:
        if intersections[0, 0] > np.mean(left[..., 0]):
            i1 = intersections[0]
            i2 = (left[-1] + right[0]) / 2
        else:
            i1 = (left[0] + right[-1]) / 2
            i2 = intersections[0]
    else:
        i1 = (left[0] + right[-1]) / 2
        i2 = (left[-1] + right[0]) / 2

    left_padding = polygon.shape[0] - left.shape[0] - 2
    right_padding = polygon.shape[0] - right.shape[0] - 2

    left_pad_lat = np.linspace(i2[0], i1[0], left_padding + 2)[1:-1]
    right_pad_lat = np.linspace(i1[0], i2[0], right_padding + 2)[1:-1]

    left_pad = np.vstack((left_pad_lat, np.full_like(left_pad_lat, i2[1]))).T
    right_pad = np.vstack((right_pad_lat, np.full_like(right_pad_lat, i2[1]))).T

    left = np.append(left, (i2, *left_pad, i1), axis=0)
    right = np.append(right, (i1, *right_pad, i2), axis=0)
    right[..., 1] -= 360

    return left, right


def _process_boundaries(
    lat: ArrayLike, lon: ArrayLike, bisection: float = 180, eps: float = 1e-6
) -> tuple[LatLons, NDArray | None, NDArray | None]:
    lat, lon = np.copy(np.asarray((lat, lon)))
    candidate_idx = _candidate_boundary_indices(lon)

    if candidate_idx.size != 0:
        candidate_lat = lat[candidate_idx]
        candidate_lon = _transform_candidate(lon[candidate_idx], eps=eps)

        accepted_idx = _accepted_candidate_indices(candidate_lon, eps=eps)
        denied_idx = np.setdiff1d(np.arange(candidate_lon.shape[0]), accepted_idx)

        denied_lat, denied_lon = candidate_lat[denied_idx], candidate_lon[denied_idx]
        accepted_lat, accepted_lon = (
            candidate_lat[accepted_idx],
            candidate_lon[accepted_idx],
        )

        split_poly = [
            _split_polygon(lat, lon, bisection=bisection, eps=eps)
            for lat, lon in zip(accepted_lat, accepted_lon)
        ]
        split_poly = np.asarray(split_poly)

        denied_lon -= 360
        lat[candidate_idx[denied_idx]] = denied_lat
        lon[candidate_idx[denied_idx]] = denied_lon

        return (lat, lon), split_poly, candidate_idx[accepted_idx]

    return (lat, lon), None, None


def _concat_boundaries(
    lat: ArrayLike, lon: ArrayLike, poly: ArrayLike, poly_idx: ArrayLike
) -> NDArray:
    boundaries = np.stack((lat, lon), axis=-1)
    poly = np.asarray(poly)
    poly_idx = np.asarray(poly_idx)

    if poly.size == 0:
        return boundaries

    base_idx = np.setdiff1d(np.arange(boundaries.shape[0]), poly_idx)
    additional = np.reshape(poly, shape=(2 * poly_idx.shape[0], -1, 2))
    return np.append(boundaries[base_idx], additional, axis=0)


def _transform_base_to_polymap(data: ArrayLike, poly_idx: ArrayLike) -> NDArray:
    data = np.copy(np.asarray(data))
    poly_idx = np.asarray(poly_idx)
    base_idx = np.setdiff1d(np.arange(data.shape[0]), poly_idx)
    additional = np.repeat(data[poly_idx], 2, axis=0)
    return np.append(data[base_idx], additional, axis=0)


def pixgrid_to_poly(
    nside: int,
    pixgrid: ArrayLike,
    cmaps: list[ArrayLike] | None = None,
    step: int = 1,
    bisection: float = 180,
    eps: float = 1e-6,
) -> NDArray | tuple[NDArray, list[NDArray]]:
    """
    Given a pixgrid and optional colormaps, returns a set of polygons
    useable as a matplotlib collection and their associated colors.
    """
    pixgrid = np.asarray(pixgrid)
    blat, blon = _pix_boundaries(nside, pixgrid, step=step)
    (plat, plon), poly, poly_idx = _process_boundaries(
        blat, blon, bisection=bisection, eps=eps
    )

    if poly is None or poly_idx is None:
        polygon = np.stack((plat, plon), axis=-1)
    else:
        polygon = _concat_boundaries(plat, plon, poly, poly_idx)

    polygon = np.flip(polygon, axis=-1)

    if cmaps is None:
        return polygon

    if poly_idx is not None:
        cmaps = [_transform_base_to_polymap(cmap, poly_idx) for cmap in cmaps]
    return polygon, cmaps  # ty: ignore
