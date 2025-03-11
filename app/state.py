import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Generic, TypeVar

from pydantic_core import from_json
from redis import Redis
from redis.typing import ResponseT

from .models import InsertInputSchema
from .wrappers import redis_kit

_t = TypeVar("T")


class BaseModelHandler(Generic[_t], ABC):
    """Implementation of base model handler."""

    base_redis_key_prefix = f"model_{_t.__name__}:"
    default_expiry_time = 60 * 60 * 24 * 7

    def __init__(
        self,
        redis_client: Redis | None = None,
    ) -> None:
        super().__init__()
        self.redis_client = (
            redis_client or redis_kit.reusable_redis_connection()
        )

    def get(self, id: str) -> _t:  # noqa: A002
        """Get item by ID."""
        return self.from_bytes(self.redis_client.get(self.key(id)))

    def insert(self, item: _t) -> ResponseT:
        """Insert item."""
        return self.redis_client.set(
            self.key(self.id(item)),
            self.to_bytes(item),
        )

    def delete(self, id: str) -> ResponseT:  # noqa: A002
        """Delete item by ID."""
        return self.redis_client.delete(self.key(id))

    def get_all(self) -> list[_t]:
        """Get all items."""
        return [
            self.from_bytes(self.redis_client.get(key))
            for key in self.keys()
        ]

    @abstractmethod
    def to_bytes(self, item: _t) -> bytes:
        """Convert to bytes.

        Mark as abstract method.

        """
        raise NotImplementedError

    def from_bytes(self, data: bytes) -> _t:
        """Handle from bytes.

        Mark as abstract method.

        """
        raise NotImplementedError

    def key(self, id: str) -> str:  # noqa: A002
        """Return key."""
        return f"{self.base_redis_key_prefix}:{id}"

    def keys(self) -> list[str]:
        """Return keys."""
        return self.redis_client.keys(f"{self.base_redis_key_prefix}*")

    def id(self, item: _t) -> str:
        """Implement ID of item."""
        return item.id if hasattr(item, "id") else hash(self.to_bytes(item))


class InsertionRequestHandler(BaseModelHandler[InsertInputSchema]):
    """Implementation of insertion request handler."""

    def to_bytes(self, item: InsertInputSchema) -> bytes:
        """Convert to bytes."""
        return json.dumps(item.model_dump()).encode("utf-8")

    def from_bytes(self, data: bytes) -> InsertInputSchema:
        """Handle from bytes."""
        return InsertInputSchema.model_validate(from_json(data))


@lru_cache(maxsize=1)
def get_insertion_request_handler() -> InsertionRequestHandler:
    """Return insertion request handler."""
    return InsertionRequestHandler()
