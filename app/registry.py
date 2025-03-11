import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

__registry = {}


class ClassRegistry(str, Enum):
    """Implementation of class registry."""

    QUERY_OPTIMIZER = "query_optimizer"
    LANGUAGE_MODEL = "language_model"


def get_registered(category: ClassRegistry) -> list[str]:
    """Get registered classes."""
    global __registry
    return __registry.get(category, [])


def register(category: ClassRegistry, cls: Any) -> Any:
    """Register a class."""
    global __registry

    if category not in __registry:
        __registry[category] = []

    if not hasattr(cls, '__name__'):
        logger.error(f"Class {cls} does not have __name__ attribute")
        return False

    logger.info(f"Registering {cls.__name__} as a {category}")
    __registry[category].append(cls)
    return True


def register_decorator(category: ClassRegistry) -> Any:
    """Register a class."""
    def decorator(cls: Any) -> Any:
        register(category, cls)
        return cls
    return decorator


def get_cls(category: ClassRegistry, name: str) -> Any:
    """Implement get class."""
    global __registry

    for cls in __registry.get(category, []):
        if cls.__name__ == name:
            return cls

    return None
