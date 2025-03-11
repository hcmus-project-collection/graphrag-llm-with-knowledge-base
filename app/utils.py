import asyncio
import logging
import os
import time
import traceback
from asyncio import Semaphore as AsyncSemaphore
from collections.abc import AsyncGenerator, Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps
from hashlib import md5
from pathlib import Path
from typing import Any

import aiofiles
from pymilvus import Collection, CollectionSchema
from starlette.concurrency import run_in_threadpool

from app.models import EmbeddingModel, SimMetric

logger = logging.getLogger(__name__)


class FailedAfterRetry(Exception):
    """Implementation of FailedAfterRetry exception."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_content_checksum(data: bytes | str) -> str:
    """Get checksum of content."""
    _data = data.encode() if isinstance(data, str) else data

    return md5(_data, usedforsecurity=False).hexdigest()


def batching(data: Generator, batch_size: int = 1) -> Generator:
    """Implement batching."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


async def async_batching(
    data: AsyncGenerator,
    batch_size: int = 1,
) -> Generator:
    """Implement async batching."""
    current_batch = []

    async for item in data:
        current_batch.append(item)

        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []

    if len(current_batch) > 0:
        yield current_batch


def get_hash(*items) -> str:
    """Get hash of items."""
    return md5("".join(items).encode()).hexdigest()  # noqa: S324


def sync2async(sync_func: Callable) -> Callable:
    """Convert sync function to async function."""
    async def async_func(*args, **kwargs) -> Any:
        return await run_in_threadpool(partial(sync_func, *args, **kwargs))
    return (
        async_func
        if not asyncio.iscoroutinefunction(sync_func)
        else sync_func
    )


def sync2async_in_subprocess(sync_func: Callable) -> Callable:
    """Convert sync function to async function in subprocess."""
    async def async_func(*args, **kwargs) -> Any:
        wrapper = partial(sync_func, *args, **kwargs)

        with ProcessPoolExecutor(max_workers=1) as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, wrapper,
            )

    return (
        async_func
        if not asyncio.iscoroutinefunction(sync_func)
        else sync_func
    )


def limit_asyncio_concurrency(num_of_concurrent_calls: int) -> Callable:
    """Limit asyncio concurrency."""
    semaphore = AsyncSemaphore(num_of_concurrent_calls)

    def decorator(func: Callable) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def random_payload(length: int) -> str:
    """Implement random payload."""
    return os.urandom(length).hex()


def get_tmp_directory() -> Path:
    """Get temporary directory."""
    return Path.cwd() / ".tmp" / random_payload(20)


def is_async_func(func: Callable) -> bool:
    """Check if function is async."""
    return asyncio.iscoroutinefunction(func)


def background_task_error_handle(handler: Callable) -> Callable:
    """Implement background task error handler."""
    def decorator(func: Callable) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                res = handler(*args, e, **kwargs)

                if is_async_func(handler):
                    return await res

        return wrapper
    return decorator


def estimate_ip_from_distance(
    distance: float,
    model_use: EmbeddingModel,
) -> float:
    """Estimate IP from distance."""
    if model_use.prefer_metric == SimMetric.COSINE:
        return 1.0 - distance

    if model_use.prefer_metric == SimMetric.L2:
        return 1.0 / (1.0 + distance)

    return distance


async def iter_file(file_name: str) -> None:
    """Iterate over a file."""
    async with aiofiles.open(file_name, "rb") as f:
        while True:
            chunk = await f.read(1024 * 20)

            if not chunk:
                break

            yield chunk


def retry(
    func: Callable,
    max_retry: int = 5,
    first_interval: int = 10,
    interval_multiply: int = 1,
) -> Callable:
    """Retry decorator."""
    def sync_wrapper(*args, **kwargs) -> None:
        interval = first_interval
        for iteration in range(max_retry + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                traceback.print_exc()
                logger.exception(
                    f"Function {func.__name__} failed with error. "
                    f"Retry attempt {iteration}/{max_retry}",
                )

            time.sleep(interval)
            interval *= interval_multiply
        msg = f"Function {func.__name__} failed after all retry."
        logger.error(msg)
        raise FailedAfterRetry(msg)

    async def async_wrapper(*args, **kwargs) -> None:
        interval = first_interval
        for iteration in range(max_retry + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                traceback.print_exc()
                logger.exception(
                    f"Function {func.__name__} failed with error . "
                    f"Retry attempt {iteration}/{max_retry}",
                )
            await asyncio.sleep(interval)
            interval *= interval_multiply

        msg = f"Function {func.__name__} failed after all retry."
        logger.error(msg)
        raise FailedAfterRetry(msg)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def is_valid_schema(
    collection_name: str,
    required_schema: CollectionSchema,
) -> bool:
    """Check if schema is valid."""
    collection = Collection(collection_name)
    schema = collection.schema
    return schema == required_schema
