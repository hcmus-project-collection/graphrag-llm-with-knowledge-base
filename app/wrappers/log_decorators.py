import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from .telegram_kit import send_message

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_function_call(func: Callable) -> Any:
    """Log the function call.

    Logs the name of the function being called, its arguments,
    and return value.

    """

    def wrapper(*args, **kwargs) -> Any:
        logging.info(
            f"Function `{func.__name__}` called with args: {args} and "
            f"kwargs: {kwargs}",
        )
        result = func(*args, **kwargs)
        logging.info(f"Function `{func.__name__}` returned: {result}")
        return result

    return wrapper


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


def log_on_error(func: Callable) -> Any:
    """Log an error if the function raises an exception."""

    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(
                f"Function `{func.__name__}` raised an error: {e}",
                exc_info=True,
            )

            raise e  # noqa: TRY201

    return wrapper


def log_on_error_and_raise_alert(alert_room: str | None = None) -> Any:
    """Log an error if the function raises an exception."""

    def decorator(func: Callable) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(
                    f"Function `{func.__name__}` raised an error: {e}",
                    exc_info=True,
                )

                msg = (
                    f"## Function {func.__name__} raised an error: {e}; \n\n"
                    f"## Inputs: \n-args: {args}\n-kwargs {kwargs} \n\n"
                    f"## Traceback: \n\n ```bash\n{traceback.format_exc()}"
                    "\n```"
                )

                alert_room is None or send_message(
                    "junk_notifications",
                    msg,
                    room=alert_room,
                )
                raise e  # noqa: TRY201
        return wrapper
    return decorator


def log_custom_message(message: str, level: int = logging.INFO) -> Any:
    """Log a custom message before calling the function."""

    def decorator(func: Callable) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            logging.log(
                level,
                f"{message} - Function `{func.__name__}` is starting.",
            )
            result = func(*args, **kwargs)
            logging.log(
                level,
                f"{message} - Function `{func.__name__}` completed.",
            )
            return result

        return wrapper

    return decorator


def log_all(func: Callable) -> Any:
    """Combine logging of function calls, execution time, and error handling.

    Example for combining multiple decorators.

    """

    @log_function_call
    @log_execution_time
    @log_on_error
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)

    return wrapper
