from functools import lru_cache

from pymilvus import MilvusClient


@lru_cache(maxsize=128)
def get_reusable_milvus_client(uri: str) -> MilvusClient:
    """Return a reusable MilvusClient instance."""
    return MilvusClient(uri=uri)
