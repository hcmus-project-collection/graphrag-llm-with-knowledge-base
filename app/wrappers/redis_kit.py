import asyncio
import logging
import os
import pickle
from collections.abc import Callable
from functools import wraps
from typing import Any

import redis
import redis.commands

logger = logging.getLogger(__name__)


class RedisConnectioNotAlive(Exception):
    """Exception for Redis connection not alive."""

    def __init__(self, message: str = "Redis connection is not alive") -> None:
        super().__init__(message)


def get_redis_client_connection() -> redis.Redis:
    """Get a Redis connection."""
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=os.getenv("REDIS_PORT", "6379"),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", ""),
    )


__redis_connection = None


def reusable_redis_connection(strict: bool = False) -> redis.Redis | None:
    """Return a reusable Redis connection.

    Check if the connection is still alive and reconnect if not.

    """
    global __redis_connection

    if __redis_connection is None:
        __redis_connection = get_redis_client_connection()

    still_alive = False
    for _ in range(3):
        try:
            __redis_connection.ping()
            still_alive = True
            break
        except Exception:
            __redis_connection = get_redis_client_connection()

    if not still_alive:
        if strict:
            raise RedisConnectioNotAlive()

        return None

    return __redis_connection


def get_parameters_hash(*args, **kwargs) -> int:
    """Get hash of the parameters."""
    return hash((*args, *sorted(kwargs.items())))


def generate_cache_key(fn_module: str, fn_name: str, *args, **kwargs) -> str:
    """Generate a unique cache key based on function and parameters."""
    base_key = f"{fn_module}.{fn_name}:result"
    param_hash = get_parameters_hash(*args, **kwargs)
    return f"{base_key}:{param_hash}"


def get_redis_client() -> redis.Redis | None:
    """Get a reusable Redis connection."""
    return reusable_redis_connection()


async def get_from_cache(key: str) -> Any:
    """Retrieve a cached value from Redis."""
    redis_client = get_redis_client()
    if redis_client:
        pickle_str = redis_client.get(key)
        if pickle_str is not None:
            logger.info(f"Cache hit for {key}")
            return pickle.loads(pickle_str)  # noqa: S301
    return None


async def save_to_cache(key: str, value: Any, interval_seconds: float) -> None:
    """Save a value to Redis with an expiration time."""
    redis_client = get_redis_client()
    if redis_client:
        pickle_str = pickle.dumps(value)
        redis_client.set(key, pickle_str, ex=interval_seconds)


def cache_for(interval_seconds: float) -> Callable:
    """Implement decorator to cache function results in Redis."""
    def decorator(func: Callable) -> Callable:
        fn_name = func.__name__
        fn_module = func.__module__

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            key = generate_cache_key(fn_module, fn_name, *args, **kwargs)
            cached_value = asyncio.run(get_from_cache(key))

            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            asyncio.run(save_to_cache(key, result, interval_seconds))
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            key = generate_cache_key(fn_module, fn_name, *args, **kwargs)
            cached_value = await get_from_cache(key)

            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)
            await save_to_cache(key, result, interval_seconds)
            return result

        return (
            async_wrapper
            if asyncio.iscoroutinefunction(func)
            else sync_wrapper
        )

    return decorator
