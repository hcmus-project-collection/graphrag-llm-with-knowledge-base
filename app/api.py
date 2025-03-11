import logging
import shutil
from functools import partial
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.state import get_insertion_request_handler

from .embedding import get_default_embedding_model
from .handlers import drop_kb, get_sample, process_data, run_query
from .io import export_collection_data
from .models import (
    APIStatus,
    InsertInputSchema,
    QueryInputSchema,
    QueryResult,
    ResponseMessage,
    UpdateInputSchema,
)
from .utils import get_tmp_directory, iter_file

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)


@router.post("/insert", response_model=ResponseMessage[str])
async def insert(
    request: InsertInputSchema,
    background_tasks: BackgroundTasks,
) -> ResponseMessage[str]:
    """Implement logic for inserting documents."""
    print(f"REQUEST: {request}")
    handler = get_insertion_request_handler()
    handler.insert(request)

    background_tasks.add_task(
        process_data,
        request,
        get_default_embedding_model(),
    )

    return ResponseMessage[str](
        result="successfully submitted documents",
        status=APIStatus.OK,
    )


@router.post("/update", response_model=ResponseMessage[str])
async def update(
    request: UpdateInputSchema,
    background_tasks: BackgroundTasks,
) -> ResponseMessage[str]:
    """Implement logic for updating documents."""
    handler = get_insertion_request_handler()
    handler.insert(request)

    background_tasks.add_task(
        process_data,
        request,
        get_default_embedding_model(),
    )

    return ResponseMessage[str](
        result="successfully submitted documents",
        status=APIStatus.OK,
    )


@router.post("/query", response_model=ResponseMessage[list[QueryResult]])
async def query(
    request: QueryInputSchema,
    background_tasks: BackgroundTasks,
) -> ResponseMessage[list[QueryResult]]:
    """Implement query endpoint."""
    return ResponseMessage[list[QueryResult]](result=await run_query(request))


@router.get("/sample", response_model=ResponseMessage[list[QueryResult]])
async def sample(kb: str, k: int) -> ResponseMessage[list[QueryResult]]:
    """Implement sample logic."""
    return ResponseMessage[list[QueryResult]](result=await get_sample(kb, k))


@router.delete("/delete", response_model=ResponseMessage[str])
async def delete(
    kb: str,
    background_tasks: BackgroundTasks,
) -> ResponseMessage[str]:
    """Delete all documents in a knowledge base."""

    return ResponseMessage[str](
        result=f"{await drop_kb(kb)} documents deleted",
    )


@router.get(
    "/stat",
    response_model=ResponseMessage[str],
    include_in_schema=False,
)
async def stat() -> ResponseMessage[str]:
    """Return the stat."""
    return ResponseMessage[str](result="OK")


@router.get(
    "/progress",
    response_model=ResponseMessage[str],
    include_in_schema=False,
)
async def stat() -> ResponseMessage[str]:  # noqa: F811
    """Return the stat."""
    return ResponseMessage[str](result="OK")


@router.get("/export", include_in_schema=False)
async def export(
    collection_name: str,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """Export collection data."""
    ws = get_tmp_directory()
    Path(ws).mkdir(parents=True, exist_ok=True)

    shutil_rmtree = partial(shutil.rmtree, ws, ignore_errors=True)

    background_tasks.add_task(shutil_rmtree)

    result_file = await export_collection_data(
        collection_name, ws,
        include_embedding=False,
        include_identity=True,
    )

    file_name = Path(result_file).name

    return StreamingResponse(
        iter_file(result_file),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={file_name}",
        },
    )
