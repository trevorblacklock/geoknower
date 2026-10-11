import asyncio
import base64
import hashlib
import hmac
import itertools
import json
import logging
from collections.abc import Awaitable
from typing import Any
from urllib.parse import urlparse

import aiohttp
import backoff
import requests
from numpy.typing import NDArray
from tqdm.asyncio import tqdm

__all__ = [
    "batch_async_pano",
    "batch_async_pano_metadata",
    "batch_async_streetview_metadata",
    "get_streetview_session",
]

logger = logging.getLogger(__name__)


def on_backoff(backoff_event):
    logger.warning(
        "http backoff event",
        extra={
            "Retrying in, seconds": {backoff_event["wait"]},
            "Attempt number": {backoff_event["tries"]},
            "Exception": {backoff_event["exception"]},
            "Target function": {backoff_event["target"].__name__},
            "Args": {backoff_event["args"]},
            "Kwargs": {backoff_event["kwargs"]},
            "Elapsed time": {backoff_event["elapsed"]},
        },
    )


def on_giveup(giveup_event):
    logger.error(
        "http giveup event",
        extra={
            "Giving up after retries exceeded": giveup_event["tries"],
            "Exception": giveup_event["exception"],
            "Target function": giveup_event["target"].__name__,
            "Args": giveup_event["args"],
            "Kwargs": giveup_event["kwargs"],
            "Elapsed time": giveup_event["elapsed"],
        },
    )


@backoff.on_exception(
    backoff.expo,
    [aiohttp.ClientError, asyncio.TimeoutError],
    jitter=backoff.random_jitter,
    max_tries=5,
    max_time=60,
    on_backoff=on_backoff,
    on_giveup=on_giveup,
)
async def http_get(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    bytes: bool = False,
):
    async with semaphore, session.get(url) as response:
        if response.ok:
            try:
                if bytes:
                    return await response.read()
                return await response.json()
            except json.JSONDecodeError as json_decode_error:
                raise ValueError("Failed to parse response JSON") from json_decode_error
        else:
            assert response.reason is not None
            raise aiohttp.ClientResponseError(
                response.request_info,
                response.history,
                status=response.status,
                message=response.reason,
            )


def request_signature(url: str, key: str) -> str:
    parsed = urlparse(url)
    url = f"{parsed.path}?{parsed.query}"

    modified_key = key.replace("-", "+").replace("_", "/")
    padding = len(modified_key) % 4
    if padding != 0:
        modified_key = f"{modified_key}{'=' * padding}"

    raw_key = base64.b64decode(modified_key)
    signature_bytes = hmac.new(raw_key, url.encode("utf-8"), hashlib.sha1).digest()

    signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")
    modified_key = signature_b64.replace("+", "-").replace("/", "_")
    if padding != 0:
        modified_key = modified_key[:-padding]

    return modified_key


def build_url(base_url: str, params: dict[str, str]) -> str:
    new_url = base_url + "?"
    for key, val in params.items():
        new_url += f"{key}={val}&"

    return new_url[:-1]


def get_streetview_session(api_key: str) -> dict[str, Any]:
    base_url = f"https://tile.googleapis.com/v1/createSession?key={api_key}"
    payload = {"mapType": "streetview", "language": "en-US", "region": "CA"}

    response = requests.post(base_url, json=payload)
    response.raise_for_status()

    return response.json()


async def safe_fn[T](coroutine: Awaitable[T]) -> T | None:
    try:
        return await coroutine
    except (aiohttp.ClientError, TimeoutError):
        return None


async def async_pano_metadata(
    pano_id: str,
    api_key: str,
    session_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    url = (
        "https://tile.googleapis.com/v1/streetview/metadata?"
        f"session={session_key}&key={api_key}&panoId={pano_id}"
    )

    return await http_get(url, session, semaphore, bytes=False)


async def async_get_tile(
    pano_id: str,
    z: int,
    x: int,
    y: int,
    api_key: str,
    session_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> bytes | None:
    url = (
        f"https://tile.googleapis.com/v1/streetview/tiles/{z}/{x}/{y}"
        f"?session={session_key}&key={api_key}&panoId={pano_id}"
    )

    return await safe_fn(http_get(url, session, semaphore, bytes=True))


async def async_get_pano(
    pano_id: str,
    z: int,
    api_key: str,
    session_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> list[bytes | None]:
    num_x = 2**z
    num_y = 2 ** (z - 1) if z >= 1 else 1

    coords = [(x, y) for x, y in itertools.product(range(num_x), range(num_y))]

    tasks = [
        async_get_tile(pano_id, z, x, y, api_key, session_key, session, semaphore)
        for x, y in coords
    ]
    return await asyncio.gather(*tasks)


async def async_streetview_metadata(
    lat: float,
    lng: float,
    api_key: str,
    signing_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    radius: int = 50,
    source: str = "outside",
) -> dict[str, Any] | None:
    base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    base_params = {
        "location": f"{lat},{lng}",
        "key": f"{api_key}",
        "radius": f"{radius}",
        "source": f"{source}",
    }

    signature = request_signature(build_url(base_url, base_params), signing_key)
    base_params["signature"] = signature
    url = build_url(base_url, base_params)

    result = await safe_fn(http_get(url, session, semaphore))

    if result and result.get("status") == "UNKNOWN_ERROR":
        await asyncio.sleep(0.2)
        result = await async_streetview_metadata(
            lat,
            lng,
            api_key,
            signing_key,
            session,
            semaphore,
            radius=radius,
            source=source,
        )

    return result


async def batch_async_pano_metadata(
    pano_ids: list[str],
    api_key: str,
    session_key: str,
    concurrency: int = 10,
    verbose: bool = True,
) -> list[dict[str, Any] | None]:
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_pano_metadata(
                pano_id,
                api_key,
                session_key,
                session,
                semaphore,
            )
            for pano_id in pano_ids
        ]

        if verbose:
            return await tqdm.gather(*tasks, desc="Fetching Pano Metadata")
        return await asyncio.gather(*tasks)


async def batch_async_streetview_metadata(
    lats: NDArray,
    lngs: NDArray,
    radii: NDArray,
    api_key: str,
    signing_key: str,
    source: str = "outdoor",
    concurrency: int = 10,
    verbose: bool = True,
) -> list[dict[str, Any] | None]:
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_streetview_metadata(
                lat,
                lng,
                api_key,
                signing_key,
                session,
                semaphore,
                radius=rad,
                source=source,
            )
            for lat, lng, rad in zip(lats, lngs, radii)
        ]

        if verbose:
            return await tqdm.gather(*tasks, desc="Fetching Streetview Metadata")
        return await asyncio.gather(*tasks)


async def batch_async_pano(
    pano_ids: list[str],
    z: int,
    api_key: str,
    session_key: str,
    concurrency: int = 10,
    verbose: bool = True,
) -> list[list[bytes | None]]:
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_get_pano(
                pano_id,
                z,
                api_key,
                session_key,
                session,
                semaphore,
            )
            for pano_id in pano_ids
        ]

        if verbose:
            return await tqdm.gather(*tasks, desc="Fetching Panos")
        return await asyncio.gather(*tasks)
