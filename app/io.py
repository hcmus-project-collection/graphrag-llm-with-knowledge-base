import asyncio
import json
import logging
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import httpx
import numpy as np
from aiofiles import open as aio_open  # For async file operations
from pymilvus import MilvusClient
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from app.utils import limit_asyncio_concurrency, sync2async
from app.wrappers import milvus_kit

from . import constants as const

logger = logging.getLogger(__name__)


class CollectionNotFoundError(Exception):
    """Implementation of CollectionNotFoundError."""

    def __init__(self, message: str, *args) -> None:
        super().__init__(message, *args)


class ChunkingFailedError(Exception):
    """Implementation of ChunkingFailedError."""

    def __init__(self, message: str, *args) -> None:
        super().__init__(message, *args)


async def get_milvus_client() -> MilvusClient:
    """Get Milvus client."""
    return milvus_kit.get_reusable_milvus_client(const.MILVUS_HOST)


async def ensure_collection_exists(cli: MilvusClient, collection: str) -> None:
    """Ensure collection exists."""
    if not await sync2async(cli.has_collection)(collection):
        raise CollectionNotFoundError(f"Collection {collection} not found")


def get_output_fields(
    include_embedding: bool,
    include_identity: bool,
) -> list[str]:
    """Get output fields."""
    fields = ["content", "reference", "hash", "head", "tail"]
    if include_embedding:
        fields.append("embedding")
    if include_identity:
        fields.append("kb")
    return fields


async def collect_data(
    it: Callable,
    include_embedding: bool,
    include_identity: bool,
) -> tuple[list[dict], np.ndarray]:
    """Collect data."""
    meta, vec = [], []
    hashes = set()
    scanned = 0

    while True:
        batch = await sync2async(it.next)()
        if not batch:
            break

        scanned += len(batch)
        mask, _ = deduplicate_batch(batch, hashes, include_identity)

        if include_embedding:
            vec.extend(
                [item["embedding"] for i, item in enumerate(batch) if mask[i]],
            )

        meta.extend([
            {
                "content": item["content"],
                "reference": (
                    item["reference"] if len(item["reference"]) else None
                ),
                **({"kb": item["kb"]} if include_identity else {}),
            }
            for i, item in enumerate(batch)
            if mask[i]
        ])

        logger.info(
            f"Exported {len(hashes)} (over {scanned}; "
            f"{100 * len(hashes) / scanned:.2f}%)...",
        )

    return meta, np.array(vec) if include_embedding else None


def deduplicate_batch(
    batch: list[dict],
    hashes: set,
    include_identity: bool,
) -> tuple[list[bool], set]:
    """De-duplicate batch."""
    h = [
        "{}{}{}{}".format(e["hash"], e["head"], e["tail"], e.get("kb", ""))
        for e in batch
    ]
    mask = [True] * len(batch)
    removed = 0

    for i, item in enumerate(h):
        _h = item if not include_identity else f"{item}{batch[i]['kb']}"
        if _h in hashes:
            removed += 1
            mask[i] = False
        else:
            hashes.add(_h)

    return mask, hashes


async def save_metadata(meta: list[dict], workspace_directory: str) -> None:
    """Save metadata."""
    logger.info(f"Making meta.json in {workspace_directory}")
    async with aio_open(Path(workspace_directory) / "meta.json", "w") as fp:
        await sync2async(json.dump)(meta, fp)


async def save_embeddings(vec: np.ndarray, workspace_directory: str) -> None:
    """Save embeddings."""
    logger.info(f"Making vec.npy in {workspace_directory}")
    await sync2async(np.save)(Path(workspace_directory) / "vec.npy", vec)


async def create_zip_file(
    workspace_directory: str,
    include_embedding: bool,
) -> str:
    """Create ZIP file."""
    destination_file = f"{workspace_directory}/data.zip"
    logger.info(f"Creating zip file {destination_file}")

    with zipfile.ZipFile(destination_file, 'w') as z:
        await sync2async(z.write)(
            Path(workspace_directory) / "meta.json",
            "meta.json",
        )

        if include_embedding:
            await sync2async(z.write)(
                Path(workspace_directory) / "vec.npy",
                "vec.npy",
            )

    return destination_file


def log_completion(
    destination_file: str,
    filter_expr: str,
    collection: str,
) -> None:
    """Log completion of export."""
    file_size_mb = Path(destination_file).stat().st_size / 1024 / 1024
    logger.info(
        f"Export {filter_expr} from {collection}: Done "
        f"(filesize: {file_size_mb:.2f} MB)",
    )


async def export_collection_data(
    collection: str,
    workspace_directory: str,
    filter_expr: str = "",
    include_embedding: bool = True,
    include_identity: bool = False,
) -> str:
    """Export collection data."""
    milvus_client = await get_milvus_client()
    await ensure_collection_exists(milvus_client, collection)

    fields_output = get_output_fields(include_embedding, include_identity)
    it = milvus_client.query_iterator(
        collection,
        filter=filter_expr,
        output_fields=fields_output,
        batch_size=1000 * 10,
    )

    meta, vec = await collect_data(it, include_embedding, include_identity)
    await save_metadata(meta, workspace_directory)

    if include_embedding:
        await save_embeddings(vec, workspace_directory)

    destination_file = await create_zip_file(
        workspace_directory,
        include_embedding,
    )
    log_completion(destination_file, filter_expr, collection)

    return destination_file


async def download_file_v2(url: str, save_dir: str = ".") -> Path:
    """Download file."""
    async with httpx.AsyncClient() as cli, cli.stream("GET", url) as stream:
        stream.raise_for_status()  # Raise an error for bad responses
        headers = stream.headers

        # Extract filename from Content-Disposition header if available
        content_disposition = headers.get("Content-Disposition")

        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"')
        else:
            # Otherwise, extract from URL path
            filename = Path(urlparse(url).path).name or "downloaded_file"

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        save_path = Path(save_dir) / filename

        # Write the file in binary mode
        async with aiofiles.open(save_path, "wb") as file:
            async for chunk in stream.aiter_bytes(chunk_size=8192):
                await file.write(chunk)

        return save_path


@limit_asyncio_concurrency(4)
async def download_file(
    session: httpx.AsyncClient,
    url: str,
    path: str,
) -> None:
    """Download file."""
    async with session.stream("GET", url) as response:
        response.raise_for_status()

        async with aiofiles.open(path, 'wb') as f:
            async for chunk in response.aiter_bytes(8192):
                await f.write(chunk)

    logger.info(f"Downloaded {path}")


@limit_asyncio_concurrency(4)
async def call_docling_server(
    file_path: str,
    embedding_model_name: str,
    min_chunk_size: int = const.MIN_CHUNK_SIZE,
    max_chunk_size: int = const.MAX_CHUNK_SIZE,
    retry: int = 5,
) -> list[str]:
    """Use docling package to chunk the file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Processing {file_path} with docling package...")

    try:
        # Convert document using docling
        converter = DocumentConverter()
        result = await sync2async(converter.convert)(file_path)
        doc = result.document

        # Use HybridChunker for better document-aware chunking
        # We'll use default parameters for now but could potentially configure
        # tokenizer based on embedding_model_name in the future
        chunker = HybridChunker()
        chunk_iter = chunker.chunk(dl_doc=doc)

        # Extract text from chunks and apply contextualization
        chunks = []
        for chunk in chunk_iter:
            # Use contextualize to get the metadata-enriched text
            enriched_text = chunker.contextualize(chunk=chunk)

            # Filter chunks based on size constraints
            # Note: Using character count as approximation for now
            if len(enriched_text.strip()) >= min_chunk_size:
                # If chunk is too large, we'll include it anyway as HybridChunker
                # should handle tokenization-aware splitting
                chunks.append(enriched_text.strip())

        logger.info(
            f"Successfully split {file_path} into chunks! "
            f"Total {len(chunks)} chunks.",
        )
        return chunks

    except Exception as e:
        logger.error(f"Error while generating chunks for file {file_path}: {e}")
        raise ChunkingFailedError(f"Chunking failed: {e}")
