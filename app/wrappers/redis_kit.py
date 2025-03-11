import asyncio
import logging
import os
import pickle
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import redis
import redis.commands
from redis.commands.core import Script
from redis.typing import ResponseT, ScriptTextT

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


def has_redis_connection() -> bool:
    """Check if a  Redis connection is alive."""
    try:
        redis_client = get_redis_client_connection()
        redis_client.ping()
        return True
    except Exception:
        return False


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


class IRQueue:
    """Implement of the interface for a queue."""

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self.redis_client = get_redis_client_connection()

    @property
    def qsize(self) -> Awaitable[int] | int:
        """Return the size of the queue."""
        return self.redis_client.llen(self.collection_name)

    def clear(self) -> None:
        """Clear the queue."""
        self.redis_client.delete(self.collection_name)

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.qsize == 0


class RNormalQueue(IRQueue):
    """Implementation of a normal queue using Redis."""

    def enqueue(self, data: dict | str) -> None:
        """Enqueue data into the queue."""
        self.redis_client.rpush(self.collection_name, data)

    def dequeue(self, block: bool = True) -> dict | str:
        """Dequeue data from the queue."""
        if block:
            res = self.redis_client.blpop([self.collection_name])

            if res is not None:
                res = res[1].decode("utf-8")

        else:
            res = self.redis_client.lpop(self.collection_name)

            if res is not None:
                res = res.decode("utf-8")

        return res


class RPriorityQueue(IRQueue):
    """Implementation of a priority queue using Redis."""

    DISTRIBUTED_DEQUEUE_SCRIPT = """
local item = redis.call('ZRANGE', KEYS[1], 0, 0)
if item[1] then
    redis.call('ZREM', KEYS[1], item[1])
    return item[1]
else
    return nil
end
"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dequeue_script = self.redis_client.register_script(
            self.DISTRIBUTED_DEQUEUE_SCRIPT,
        )

    def enqueue(self, data: bytes | str, priority: int) -> None:
        """Implement enqueuing into the queue."""
        self.redis_client.zadd(self.collection_name, {data: priority})

    def dequeue(self) -> Script:
        """Implement dequeuing from the queue."""
        res = self.dequeue_script(keys=[self.collection_name])
        return res

    def peek(self) -> Any:
        """Implement peeking into the queue."""
        res = self.redis_client.zrange(self.collection_name, 0, 0)
        return res[0] if res else None

    @property
    def qsize(self) -> ResponseT:
        """Implement the size of the queue."""
        return self.redis_client.zcard(self.collection_name)


class RAdvancedPriorityQueue(RPriorityQueue):
    """Implementation of a priority queue with advanced features."""

    DISTRIBUTED_DEQUEUE_SCRIPT = """
local item = redis.call('ZRANGEBYSCORE', KEYS[1], ARGV[1], ARGV[2], 'LIMIT', ARGV[3], ARGV[4], 'WITHSCORES')
if item[1] then
    redis.call('ZREM', KEYS[1], item[1])
    return item
else
    return nil
end
"""  # noqa: E501

    def __init__(self, priority_factory: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.priority_factory = priority_factory

    def enqueue(self, data: str, priority: float | None = None) -> None:
        """Enqueue data with priority."""
        if priority is None:
            priority = self.priority_factory(data)

        self.redis_client.zadd(self.collection_name, {data: priority})

    def dequeue(
        self,
        min_score: float = float("-inf"),
        max_score: float | None = None,
    ) -> tuple:
        """Deque an item with a score within the given range."""
        if max_score is None:
            max_score = self.priority_factory(float("inf"))

        res = self.dequeue_script(
            keys=[self.collection_name], args=[min_score, max_score, 0, 1],
        )

        if res is not None:
            res[1] = float(res[1])
            res[0] = res[0].decode("utf-8")

        return res

    @property
    def qsize(self) -> ResponseT:
        """Return the size of the queue."""
        return self.redis_client.zcard(self.collection_name)


def register_script(
    cli: redis.Redis,
    script: ScriptTextT,
) -> redis.commands.core.Script:
    """Register script with Redis."""
    return cli.register_script(script)


_reg = None
_atomic_script = """
local current_value = redis.call('GET', KEYS[1])
if current_value == false then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    return true
end
return false
"""


def atomic_check_and_set_flag(
    redis_client: redis.Redis, key: str, timeout: float, value: str = "1",
) -> bool:
    """Check if a key exists and set it if it doesn't."""
    global _reg, _atomic_script

    if _reg is None:
        _reg = register_script(redis_client, _atomic_script)

    return _reg(keys=[key], args=[value, int(timeout)])


def distributed_scheduling_job(
    interval_seconds: float,
    strict: bool = False,
) -> Any:
    """Schedule a job to run at intervals using Redis."""
    def decorator(func: Callable) -> Any:
        """Decorate to schedule a job to run at intervals using Redis."""
        def wrapper(*args, **kwargs) -> Any:
            """Wrap function to schedule a job to run at intervals .

            Wrap using Redis.

            """
            fn_name = func.__name__
            fn_module = func.__module__

            redis_cache_key = f"{fn_module}.{fn_name}:executed"

            redis_client = reusable_redis_connection(strict)
            do_execute = True

            if redis_client is not None:
                do_execute = atomic_check_and_set_flag(
                    redis_client,
                    redis_cache_key,
                    interval_seconds,
                )

            redis_cache_key_result = f"{fn_module}.{fn_name}:result"
            if not do_execute:

                if redis_client is not None:
                    pickle_str = redis_client.get(redis_cache_key_result)

                    if pickle_str is not None:
                        return pickle.loads(pickle_str)  # noqa: S301

                return None

            res = func(*args, **kwargs)

            if redis_client is not None:
                pickle_str = pickle.dumps(res)
                redis_client.set(
                    redis_cache_key_result,
                    pickle_str,
                    ex=interval_seconds,
                )

            return res

        return wrapper

    return decorator


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
