import asyncio
import json
import logging
import random
import shutil
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import httpx
import requests
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.io import (
    call_docling_server,
    download_file_v2,
    hook,
)
from app.utils import estimate_ip_from_distance, is_valid_schema

from . import constants as const
from .embedding import get_default_embedding_model, get_embedding_models
from .graph_handlers import Triplet, get_graph_knowledge
from .models import (
    APIStatus,
    CollectionInspection,
    EmbeddedItem,
    EmbeddingModel,
    GraphEmbeddedItem,
    InsertInputSchema,
    InsertionCounter,
    InsertProgressCallback,
    InsertResponse,
    QueryInputSchema,
    QueryResult,
    ResponseMessage,
)
from .state import get_insertion_request_handler
from .utils import (
    async_batching,
    batching,
    get_content_checksum,
    get_tmp_directory,
    limit_asyncio_concurrency,
    retry,
    sync2async,
)
from .wrappers import milvus_kit, redis_kit
from .wrappers.log_decorators import log_execution_time

logger = logging.getLogger(__name__)


@limit_asyncio_concurrency(
    const.DEFAULT_CONCURRENT_EMBEDDING_REQUESTS_LIMIT * 1.5,
)
async def mk_cog_embedding_prioritized(
    text: str | list[str],
    model_use: EmbeddingModel,
) -> list[list[float]]:
    """Make cog embedding prioritized."""
    url = model_use.base_url

    headers = {}

    if isinstance(text, str):
        text = [text]

    data = {
        'input': {
            "texts": text,
            "dimension": model_use.dimension,
        },
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url + '/predictions',
            headers=headers,
            json=data,
            timeout=httpx.Timeout(60.0 * 5),
        )

    if response.status_code != 200:
        raise ValueError(
            f"Failed to get embedding from {url}; "
            f"Reason: {response.text}",
        )

    response_json = response.json()
    return response_json['output']['result']


@limit_asyncio_concurrency(const.DEFAULT_CONCURRENT_EMBEDDING_REQUESTS_LIMIT)
async def mk_cog_embedding(
    text: str | list[str],
    model_use: EmbeddingModel,
) -> list[list[float]]:
    """Make cog embedding."""
    return await mk_cog_embedding_prioritized(text, model_use)


async def url_graph_chunking(
    url_or_texts: str,
    model_use: EmbeddingModel,
) -> AsyncGenerator:
    """Chunk the URL and construct graph."""
    gk = get_graph_knowledge()
    chunks = await call_docling_server(url_or_texts, model_use.tokenizer)

    futures = [
        asyncio.ensure_future(gk.construct_graph_from_chunk(item))
        for item in chunks
    ]

    results = await asyncio.gather(*futures, return_exceptions=True)

    for item, graph_result in zip(chunks, results, strict=False):
        graph_result: Exception | ResponseMessage[list[Triplet]]

        if (
            isinstance(graph_result, Exception)
            or graph_result.status != APIStatus.OK
        ):
            shortened_item = item[:100].replace("\n", "\\n")
            err_msg = (
                graph_result.error
                if not isinstance(graph_result, Exception)
                else graph_result
            )
            logger.error(
                f"Failed to construct graph from {shortened_item}. "
                f"Reason: {err_msg}",
            )
            yield item, None
        else:
            for triplet in graph_result.result:
                yield item, triplet


async def insert_to_collection(
    inputs: list[GraphEmbeddedItem],
    model_use: EmbeddingModel,
    metadata: dict,
) -> int:
    """Insert to collection."""
    assert all(  # noqa: S101
        k in metadata for k in ["kb", "reference"]
    ), "Missing required fields in metadata"

    logger.info(f"inserting {len(inputs)} entities to {model_use.identity()}")

    vectors = [e.embedding for e in inputs]
    raw_texts = [e.raw_text for e in inputs]
    heads = [e.head for e in inputs]
    tails = [e.tail for e in inputs]
    kb_postfixes = [e.kb_postfix for e in inputs]
    kb = metadata.pop('kb')

    futures = [
        asyncio.ensure_future(sync2async(get_content_checksum)(text))
        for text in raw_texts
    ]

    hashs = await asyncio.gather(*futures, return_exceptions=True)

    for i in range(len(hashs)):
        if isinstance(hashs[i], Exception):
            hashs[i] = "0" * 64

    data = [
        {
            **metadata,
            'kb': kb + kb_postfix,
            'head': head,
            'tail': tail,
            'content': text,
            'hash': await sync2async(get_content_checksum)(text),
            'embedding': vec,
        }
        for vec, text, head, tail, kb_postfix
        in zip(vectors, raw_texts, heads, tails, kb_postfixes, strict=False)
    ]

    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )

    res = await sync2async(cli.insert)(
        collection_name=model_use.identity(),
        data=data,
    )

    insert_cnt = res['insert_count']
    logger.info(
        f"Successfully inserted {insert_cnt} items to {kb} "
        f"(collection: {model_use.identity()});",
    )
    return insert_cnt


mk_cog_embedding_retry_wrapper = retry(
    mk_cog_embedding,
    max_retry=2,
    first_interval=2,
    interval_multiply=2,
)

mk_cog_embedding_retry_wrapper_prioritized = retry(
    mk_cog_embedding_prioritized,
    max_retry=2,
    first_interval=2,
    interval_multiply=2,
)


async def embed_normal_text(
    chunks: list[str],
    model_use: EmbeddingModel,
) -> AsyncGenerator:
    """Embed normal text."""
    global mk_cog_embedding_retry_wrapper

    if len(chunks) == 0:
        return

    for sub_chunks in batching(chunks, 16):
        chunks_e = await mk_cog_embedding_retry_wrapper(
            sub_chunks,
            model_use,
        )

        for chunk, e in zip(sub_chunks, chunks_e, strict=False):
            yield GraphEmbeddedItem(
                embedding=e,
                raw_text=chunk,
                kb_postfix="",
                head=0,
                tail=0,
            )


async def embed_triplet(
    chunk: str,
    triplet: Triplet,
    model_use: EmbeddingModel,
) -> tuple | None:
    """Embed a triplet."""
    global mk_cog_embedding_retry_wrapper

    relation = triplet.fact()
    head_e, tail_e, relation_e, raw_e = await mk_cog_embedding_retry_wrapper(
        [triplet.s1, triplet.s2, relation, chunk],
        model_use,
    )

    head_h, tail_h = hash(triplet.s1), hash(triplet.s2)

    return (
        GraphEmbeddedItem(
            embedding=head_e,
            raw_text=chunk,
            kb_postfix=const.ENTITY_SUFFIX,
            head=head_h,
            tail=tail_h,
        ),
        GraphEmbeddedItem(
            embedding=tail_e,
            raw_text=chunk,
            kb_postfix=const.ENTITY_SUFFIX,
            head=tail_h,
            tail=head_h,
        ),
        GraphEmbeddedItem(
            embedding=relation_e,
            raw_text=chunk,
            kb_postfix=const.RELATION_SUFFIX,
            head=head_h,
            tail=tail_h,
        ),
        GraphEmbeddedItem(
            embedding=raw_e,
            raw_text=chunk,
            kb_postfix="",
            head=0,
            tail=0,
        ),
    )


async def process_url_input(
    url: str,
    model_use: EmbeddingModel,
) -> tuple[list, list]:
    """Process a single URL, chunk and embed it."""
    futures = []
    failed = []

    async for chunk, triplet in url_graph_chunking(url, model_use):
        if triplet is not None:
            futures.append(
                asyncio.ensure_future(
                    embed_triplet(chunk, triplet, model_use),
                ),
            )
        else:
            failed.append(chunk)

    return futures, failed


async def process_text_input(
    texts: list[str],
    model_use: EmbeddingModel,
) -> tuple[list, list]:
    """Process a list of texts, construct graphs, and embed them."""
    futures = []
    failed = []
    gk = get_graph_knowledge()

    for item in texts:
        resp = await gk.construct_graph_from_chunk(item)

        if resp.status != APIStatus.OK:
            logger.error(
                f"Failed to get embedding for {item[:100] + '...'!r} "
                f"Reason: {resp.error}",
            )
            failed.append(item)
        else:
            futures.extend([
                asyncio.ensure_future(embed_triplet(item, e, model_use))
                for e in resp.result
            ])

    return futures, failed


async def handle_completed_futures(
    futures: list,
    counter: InsertionCounter,
) -> AsyncGenerator:
    """Yield results from completed futures and handle exceptions."""
    for future in asyncio.as_completed(futures):
        try:
            item = await future
            for element in item:
                yield element
        except Exception:
            counter.fails += 1
            logger.exception("Exception raised while embedding triplet")


async def handle_failed_items(
    failed: list[str],
    model_use: EmbeddingModel,
) -> AsyncGenerator:
    """Embed failed text items and yield them."""
    async for item in embed_normal_text(failed, model_use):
        yield item


async def chunking_and_embedding(
    url_or_texts: str | list[str],
    model_use: EmbeddingModel,
    counter: InsertionCounter | None = None,
) -> AsyncGenerator:
    """Implement logic for chunking and embedding."""
    counter = counter or InsertionCounter()
    futures, failed = [], []

    if isinstance(url_or_texts, str):
        futures, failed = await process_url_input(url_or_texts, model_use)
    elif isinstance(url_or_texts, list):
        futures, failed = await process_text_input(url_or_texts, model_use)
    else:
        raise TypeError(
            f"Invalid input type; Expecting str or list of str, "
            f"got {type(url_or_texts)}",
        )

    counter.total = len(futures) * 4 + len(failed)

    async for item in handle_completed_futures(futures, counter):
        yield item

    async for item in handle_failed_items(failed, model_use):
        yield item


_running_tasks = set()


async def smaller_task(
    url_or_texts: list[str] | str,
    kb: str,
    model_use: EmbeddingModel,
    file_identifier: str = "",
) -> tuple[int, int]:
    """Implement logic for smaller task."""
    if isinstance(url_or_texts, str):
        await hook(
            ResponseMessage[InsertProgressCallback](
                result=InsertProgressCallback(
                    kb=kb,
                    identifier=file_identifier,
                    message=f"Start processing file {file_identifier}",
                ),
                status=APIStatus.PROCESSING,
            ),
        )

    counter = InsertionCounter()
    async for data in async_batching(
        chunking_and_embedding(
            url_or_texts,
            model_use,
            counter,
        ),
        const.DEFAULT_MILVUS_INSERT_BATCH_SIZE,
    ):
        data: list[EmbeddedItem]

        inserted = await insert_to_collection(
            inputs=data,
            model_use=model_use,
            metadata={
                "kb": kb,
                "reference": file_identifier,
            },
        )

        counter.fails += len(data) - inserted

    logger.info(
        f"Total: {counter.total} (chunks); "
        f"Fail: {counter.fails} (chunks)",
    )

    if isinstance(url_or_texts, str) and request_identifier is not None:
        n_inserted_chunks = counter.total - counter.fails

        status = APIStatus.OK if n_inserted_chunks > 0 else APIStatus.ERROR
        reason = (
            ""
            if status == APIStatus.OK
            else "No data read from the provided file"
        )

        await hook(
            ResponseMessage[InsertProgressCallback](
                result=InsertProgressCallback(
                    message=(
                        f"Completed processing file {file_identifier}"
                        if n_inserted_chunks > 0 else
                        (
                            f"Failed to process file {file_identifier} "
                            f"(Reason: {reason})"
                        )
                    ),
                    kb=kb,
                    identifier=file_identifier,
                ),
                status=status,
            ),
        )

    return (counter.total, counter.fails)


async def inspect_by_file_identifier(
    file_identifier: str,
) -> CollectionInspection:
    """Inspect by file identifier."""
    milvus_cli = milvus_kit.get_reusable_milvus_client(const.MILVUS_HOST)

    it = milvus_cli.query_iterator(
        collection_name=get_default_embedding_model().identity(),
        filter=f"reference == {file_identifier!r}",
        output_fields=["hash"],
        batch_size=1000 * 10,
    )

    hashs = set()

    while True:
        batch = await sync2async(it.next)()

        if len(batch) == 0:
            break

        for item in batch:
            hashs.add(item['hash'])

    return CollectionInspection(
        file_ref=file_identifier,
        message=f"Has {len(hashs)} unique chunk(s)",
        status=APIStatus.OK if len(hashs) > 0 else APIStatus.ERROR,
    )


def setup_temp_directory() -> str:
    """Set up temporary directory."""
    tmp_dir = get_tmp_directory()
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    return tmp_dir


def log_request_info(req: InsertInputSchema) -> None:
    """Log request information."""
    verbosed_info = {
        k: v if k not in ["texts", "file_urls"] else f"List of {len(v)} items"
        for k, v in req.model_dump().items()
    }
    logger.info(
        f"Received {json.dumps(verbosed_info, indent=2)}; "
        f"Start handling task: {req.id}",
    )


def prepare_tasks(
    req: InsertInputSchema,
    tmp_dir: str,
    model_use: EmbeddingModel,
) -> tuple[list, list]:
    """Prepare tasks."""
    futures = []
    identifiers = []

    # Text chunks
    sqrt_length_texts = int(len(req.texts) ** 0.5)
    for chunk in batching(req.texts, sqrt_length_texts):
        futures.append(  # noqa: PERF401
            create_task(chunk, req.kb, model_use, "", req.ref),
        )

    for url in req.file_urls:
        try:
            local_filepath = asyncio.ensure_future(
                download_file_v2(url, tmp_dir),
            )
            awaitable_task = create_task(
                local_filepath,
                req.kb,
                model_use,
                url,
                req.ref,
            )
            futures.append(awaitable_task)
            identifiers.append(url)
        except Exception:
            logger.exception(f"Failed to download {url} to read locally")

    return futures, identifiers


def create_task(data, kb, model_use, file_identifier):  # noqa
    return asyncio.ensure_future(
        smaller_task(
            data,
            kb,
            model_use,
            file_identifier=file_identifier,
            request_identifier=request_identifier,
        ),
    )


async def execute_tasks(futures) -> tuple[int, int]:  # noqa: ANN001
    """Execute tasks."""
    n_chunks, fails_count = 0, 0

    if futures:
        results = await asyncio.gather(*futures, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Subtask {i} failed with error {result}")
            else:
                total, fails = result
                n_chunks += total
                fails_count += fails

    return n_chunks, fails_count


async def send_hook_if_needed(
    req: InsertInputSchema,
    n_chunks: int,
    fails_count: int,
    identifiers: Sequence,
) -> None:
    """Send hook if needed."""
    if req.hook:
        response = ResponseMessage[InsertResponse](
            result=InsertResponse(
                ref=req.ref,
                message=(
                    f"Inserted {n_chunks - fails_count} chunks; "
                    f"Failed {fails_count} chunks; "
                    f"Total: {n_chunks} chunks; "
                    f"{len(req.file_urls) + len(identifiers)} files."
                ),
                kb=req.kb,
                details=[
                    await inspect_by_file_identifier(identifier)
                    for identifier in identifiers
                ],
            ),
            status=(
                APIStatus.OK
                if (n_chunks - fails_count) > 0
                else APIStatus.ERROR
            ),
        )
        hook_result = await hook(response)
        logger.info(f"Hook status: {hook_result}")


async def cleanup_request(task_id: str) -> None:
    """Cleanup request."""
    await sync2async(get_insertion_request_handler().delete)(task_id)


def cleanup_temp_directory(tmp_dir: str, task_id: str) -> None:
    """Cleanup temporary directory."""
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Completed handling task: {task_id}")


@limit_asyncio_concurrency(4)
async def process_data(
    req: InsertInputSchema,
    model_use: EmbeddingModel,
) -> int:
    """Process data."""
    if req.id in _running_tasks:
        return 0

    tmp_dir = setup_temp_directory()
    _running_tasks.add(req.id)

    try:
        log_request_info(req)
        futures, identifiers = prepare_tasks(
            req,
            tmp_dir,
            model_use,
        )

        n_chunks, fails_count = await execute_tasks(futures)

        await send_hook_if_needed(
            req,
            n_chunks,
            fails_count,
            identifiers,
        )
        await cleanup_request(req.id)
        return n_chunks

    finally:
        _running_tasks.remove(req.id)
        cleanup_temp_directory(tmp_dir, req.id)


def resume_pending_tasks() -> None:
    """Resume pending tasks."""
    if len(_running_tasks) > 0:
        return

    logger.info("Scanning for pending tasks...")
    handler = get_insertion_request_handler()

    logger.info(f"Found {len(handler.get_all())} pending tasks")
    pending_tasks = handler.get_all()

    for task in pending_tasks[::-1]:
        if task.id in _running_tasks:
            continue

        # TODO: change this  # noqa: TD002
        logger.info(f"Resuming task {task.id}")
        requests.post(
            "http://localhost:8000/api/insert",
            json={
                **task.model_dump(),
                "is_re_submit": True,
            },
            timeout=5,
        )


async def get_collection_num_entities(collection_name: str) -> int:
    """Get number of entities in a collection."""
    cli = milvus_kit.get_reusable_milvus_client(const.MILVUS_HOST)
    res = await sync2async(cli.query)(
        collection_name=collection_name,
        output_fields=["count(*)"],
    )
    return res[0]["count(*)"]


def prepare_milvus_collection() -> None:
    """Prepare Milvus collections."""
    models = get_embedding_models()
    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )

    logger.info(f"Checking and creating collections for {len(models)} models")

    for model in models:
        identity = model.identity()
        collection_schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=1024 * 8,
                ),
                FieldSchema(
                    name="hash",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),
                FieldSchema(name="head", dtype=DataType.INT64, Default=-1),
                FieldSchema(
                    name="tail",
                    dtype=DataType.INT64,
                    Default=-1,
                ),
                FieldSchema(
                    name="reference",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                ),
                FieldSchema(
                    name="kb",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=model.dimension,
                ),
            ],
        )

        if cli.has_collection(identity):
            if is_valid_schema(identity, collection_schema):
                logger.info(f"Collection {model.identity()} is ready")
                continue
            else:
                logger.info(
                    f"Collection {model.identity()} has invalid schema. "
                    "Dropping it",
                )
                cli.drop_collection(identity)

        index_params = MilvusClient.prepare_index_params(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type=model.prefer_metric.value,
            nlist=128,
        )

        cli.create_collection(
            collection_name=model.identity(),
            schema=collection_schema,
            index_params=index_params,
        )

        logger.info(f"Collection {model.identity()} created")

    logger.info("All collections are ready")


def deduplicate_task() -> None:
    """De-duplicate the data in the collections."""
    models = get_embedding_models()
    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )
    fields_output = ["hash", "id", "kb", "head", "tail", "reference"]

    for model in models:
        identity = model.identity()

        if not cli.has_collection(identity):
            logger.error(f"Collection {identity} not found")
            continue

        first_observation = {}
        to_remove_ids = []

        it = cli.query_iterator(
            identity,
            output_fields=fields_output,
            batch_size=1000 * 10,
        )

        while True:
            batch = it.next()

            if len(batch) == 0:
                break

            for item in batch:
                item_key = "{hash}_{head}_{tail}_{ref}_{kb}".format(
                    hash=item["hash"],
                    kb=item["kb"],
                    head=item["head"],
                    tail=item["tail"],
                    ref=item["reference"],
                )

                if item_key not in first_observation:
                    first_observation[item_key] = item

                else:
                    to_remove_ids.append(item["id"])

        if len(to_remove_ids) > 0:
            logger.info(
                f"Removing {len(to_remove_ids)} duplications in {identity}",
            )
            cli.delete(
                collection_name=identity,
                ids=to_remove_ids,
            )

        logger.info(f"Deduplication for {identity} done")


@redis_kit.cache_for(interval_seconds=300 // 5)  # seconds
async def get_sample(kb: str, k: int) -> list[QueryResult]:
    """Get sample."""
    if k <= 0:
        return []

    fields_output = ["content", "reference", "hash"]

    embedding_model = get_default_embedding_model()
    model_identity = embedding_model.identity()
    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )

    relational_kb = kb  # + const.RELATION_SUFFIX

    results = await sync2async(cli.query)(
        model_identity,
        filter=f"kb == {relational_kb!r}",
        output_fields=fields_output,
    )

    results = list({
        item["hash"]: item
        for item in results
    }.values())

    results_random_k = random.sample(results, min(k, len(results)))

    return [
        QueryResult(
            content=item['content'],
            reference=item['reference'],
            score=1,
        )
        for item in results_random_k
    ]


async def drop_kb(kb: str) -> int:
    """Drop all data from a given knowledge base."""
    models = get_embedding_models()
    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )

    removed_count = 0

    for model in models:
        identity = model.identity()

        if not cli.has_collection(identity):
            logger.error(f"Collection {identity} not found")
            continue

        resp: dict = cli.delete(
            collection_name=identity,
            filter=f"kb == {kb!r}",
        )

        removed_count += resp['delete_count']

    logger.info(f"Deleted all data for kb {kb}")
    return removed_count


@log_execution_time
@redis_kit.cache_for(interval_seconds=300 // 5)  # seconds
async def run_query(req: QueryInputSchema) -> list[QueryResult]:
    """Run a query against the embedding models."""
    if len(req.kb) == 0 or req.top_k <= 0:
        return []

    embedding_model = get_default_embedding_model()
    model_identity = embedding_model.identity()

    logger.info(
        f"Searching for: {req.query!r} from {model_identity} "
        f"[kbs={req.kb}; top_k={req.top_k}; threshold={req.threshold}]",
    )

    entity_kb = [
        kb + const.ENTITY_SUFFIX
        for kb in req.kb
    ]

    relational_kb = [
        kb + const.RELATION_SUFFIX
        for kb in req.kb
    ]

    nodes = []

    # Extract named entities from the query
    resp = await get_graph_knowledge().extract_named_entities(req.query)
    logger.info(f"NER: {resp.result}")

    if resp.status != APIStatus.OK:
        logger.warning(
            "No entities extracted from the given query. "
            f"Message: {resp.error}",
        )

    ner_query_list = resp.result or []

    embeddings = await mk_cog_embedding_retry_wrapper_prioritized(
        [req.query, *ner_query_list], embedding_model,
    )

    cli: MilvusClient = milvus_kit.get_reusable_milvus_client(
        const.MILVUS_HOST,
    )

    if len(ner_query_list) > 0:
        res = await sync2async(cli.search)(
            collection_name=model_identity,
            data=embeddings[1:],
            kb_filter=f"kb in {entity_kb}",
            anns_field="embedding",
            output_fields=["head", "tail"],
            search_params={"params": {"radius": req.threshold}},
        )

        for ee in res:
            for e in ee:
                nodes.extend([
                    e['entity']['head'],
                    e['entity']['tail'],
                ])

    filter_str = f"kb in {relational_kb}"

    if len(nodes) > 0:
        nodes = list(set(nodes))
        filter_str += f" and (head in {nodes} or tail in {nodes})"

    filter_str = f"({filter_str}) or kb in {req.kb}"
    query_embedding = embeddings[0]

    res = await sync2async(cli.search)(
        collection_name=model_identity,
        data=[query_embedding],
        filter=filter_str,
        limit=max(req.top_k, 1),
        anns_field="embedding",
        output_fields=["id", "content", "reference", "hash"],
        search_params={"params": {"radius": req.threshold}},
    )

    hits = list(
        {
            item['entity']['hash']: item
            for item in res[0]
        }.values(),
    )

    for i in range(len(hits)):
        hits[i]['score'] = estimate_ip_from_distance(
            hits[i]['distance'],
            embedding_model,
        )

    hits = sorted(
        hits,
        key=lambda e: e['score'],
        reverse=True,
    )

    return [
        QueryResult(
            content=hit['entity']['content'],
            reference=hit['entity']['reference'],
            score=hit['score'],
        )
        for hit in hits
    ]


