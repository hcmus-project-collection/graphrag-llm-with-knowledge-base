
from . import constants as const
from .models import EmbeddingModel, SimMetric


def get_embedding_models() -> list[EmbeddingModel]:
    """Get embedding models."""
    return [
        EmbeddingModel(
            name=const.SELF_HOSTED_EMBEDDING_MODEL_ID,
            base_url=const.SELF_HOSTED_EMBEDDING_URL,
            tokenizer=const.TOKENIZER_MODEL_ID,
            dimension=const.MODEL_DIMENSION,
            prefer_metric=SimMetric.IP,
            normalize=False,
        ),
    ]


def get_default_embedding_model() -> EmbeddingModel:
    """Get default embedding model."""
    return get_embedding_models()[0]
