import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_execution_time(func: Callable) -> Any:
    """Log the execution time of the function."""

    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(
            f"Function `{func.__name__}` executed in "
            f"{elapsed_time:.4f} seconds",
        )
        return result

    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(
            f"Function `{func.__name__}` executed in "
            f"{elapsed_time:.4f} seconds",
        )
        return result

    return wrapper if not asyncio.iscoroutinefunction(func) else async_wrapper
