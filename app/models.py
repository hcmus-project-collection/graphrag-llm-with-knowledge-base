import string
import uuid
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

_generic_type = TypeVar("_generic_type")


class EmbeddedItem(BaseModel):
    """Implementation of embedded item."""

    embedding: list[float] | None = None
    raw_text: str
    error: str | None = None


class GraphEmbeddedItem(EmbeddedItem):
    """Implementation of graph embedded item."""

    head: int
    tail: int
    kb_postfix: str


class APIStatus(str, Enum):
    """Implementation of API status."""

    OK = "ok"
    ERROR = "error"

    PENDING = "pending"
    PROCESSING = "processing"


class InsertInputSchema(BaseModel):
    """Model for representation of insert input schema."""

    id: str = Field(default_factory=lambda: f"doc-{uuid.uuid4().hex!s}")
    file_urls: list[str] = []
    texts: list[str] = []
    kb: str | None = None

    is_re_submit: bool = False

    @model_validator(mode="before")
    def fill_texts(cls, data: dict):
        """Fill data into texts."""
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")

        if "texts" not in data:
            data["texts"] = []

        if isinstance(data["texts"], str):
            data["texts"] = [data["texts"]]

        if data.get("kb", "") == "" and "ref" not in data:
            raise ValueError(
                "Either a reference or a knowledge base must be provided",
            )

        if data.get("kb", "") == "":
            data["kb"] = "kb-" + data["ref"]

        if len(data["kb"]) == 0:
            raise ValueError("Knowledge base must not be empty")
        return data


class UpdateInputSchema(BaseModel):
    """Model for representation of update input schema."""

    kb: str | None
    id: str = Field(default_factory=lambda: f"doc-{uuid.uuid4().hex!s}")
    file_urls: list[str] = []
    texts: list[str] = []

    is_re_submit: bool = False

    @model_validator(mode="before")
    def fill_texts(cls, data: dict):
        """Fill data into texts."""
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")

        if "texts" not in data:
            data["texts"] = []

        if isinstance(data["texts"], str):
            data["texts"] = [data["texts"]]

        return data


class CollectionInspection(BaseModel):
    """Model for representation of collection inspection."""

    file_ref: str  # {cid}/{file_index}
    status: APIStatus = APIStatus.OK
    message: str = ""


class QueryInputSchema(BaseModel):
    """Model for representation of query input schema."""

    query: str
    top_k: int = 1
    kb: list[str]
    threshold: float = 0.2

    def __hash__(self) -> int:
        """Return the hash of the model."""
        kbs_str = "".join(sorted(self.kb))
        return hash(f"{self.query}{self.top_k}{self.threshold}{kbs_str}")

    @model_validator(mode="before")
    def fill_kb(cls, data: dict):
        """Fill data into knowledge base."""
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")

        if "kb" not in data:
            raise ValueError("Knowledge base must be provided")

        if isinstance(data["kb"], str):
            data["kb"] = [data["kb"]]

        return data


class SimMetric(str, Enum):
    """Implementation of similarity metric."""

    L2 = "L2"
    IP = "IP"
    COSINE = "COSINE"


class EmbeddingModel(BaseModel):
    """Model for representation of embedding model."""

    name: str
    tokenizer: str
    base_url: str
    dimension: int
    prefer_metric: SimMetric | None = SimMetric.COSINE
    normalize: bool = False

    def __hash__(self) -> int:
        """Return the hash of the model."""
        data = f"{self.name}-{self.dimension}"
        return hash(data)

    def identity(self):
        """Return the identity of the model."""
        punctuation = string.punctuation.replace("_", "")

        name = self.name.lower()
        for p in punctuation:
            name = name.replace(p, "_")

        return f"{name}_{self.dimension}"


class SearchRequest(BaseModel):
    """Model for representation of search request."""

    collection: str
    query: str
    top_k: int = 1

    kwargs: dict[str, Any] | None = None


class ResponseMessage(BaseModel, Generic[_generic_type]):
    """Model for representation of response message."""

    result: _generic_type = None
    error: str | None = None
    status: APIStatus = APIStatus.OK

    @model_validator(mode="after")
    def refine_status(self):
        """Refine the status."""
        if self.error is not None:
            self.status = APIStatus.ERROR

        return self


class InsertionCounter:
    """Implementation of insertion counter."""

    def __init__(self) -> None:
        self.total = 0
        self.fails = 0


class QueryResult(BaseModel):
    """Model for representation of query result."""

    content: str
    score: float
    reference: str | None = None
