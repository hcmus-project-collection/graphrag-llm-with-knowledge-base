import asyncio
import logging
import os
import threading
import time
from typing import Annotated

import schedule
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pymilvus import connections

from app import __version__
from app import constants as const

logging.basicConfig(level=logging.DEBUG if __debug__ else logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SECRET_TOKEN = os.environ.get("API_SECRET_TOKEN", "")


TokenType = Annotated[str | None, Header()]


def verify_token(x_token: TokenType = None) -> TokenType:
    """Verify token."""
    if x_token != SECRET_TOKEN:
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token


class EndpointFilter(logging.Filter):
    """Implement endpoint filter for logging."""

    def filter(self, record) -> bool:  # noqa: ANN001
        """Exclude specific endpoints from logging."""
        excluded_endpoints = ["GET / HTTP"]
        if any(
            endpoint in record.getMessage()
            for endpoint in excluded_endpoints
        ):
            return False
        return True


def scheduler_job() -> None:
    """Implement job scheduler."""
    if "SKIP_SCHEDULED_TASK" in os.environ:
        return

    logger.info("Scheduler started....")

    for job in schedule.default_scheduler.jobs:
        try:
            logger.info(f"Registered job: {job}")
            schedule.default_scheduler._run_job(job)
        except Exception:
            logger.exception("Scheduler error")

    while True:
        try:
            schedule.run_pending()
        except Exception:
            logger.exception("Scheduler error")
        finally:
            time.sleep(1)


# Custom logging configuration
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "endpoint_filter": {
            "()": EndpointFilter,
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "filters": ["endpoint_filter"],
        },
    },
    "loggers": {
        "uvicorn.access": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

if __name__ == "__main__":
    connections.connect(uri=const.MILVUS_HOST)

    from app.api import router as app_router
    from app.handlers import (
        deduplicate_task,
        prepare_milvus_collection,
        resume_pending_tasks,
    )
    prepare_milvus_collection()

    schedule.every(300).minutes.do(deduplicate_task)
    schedule.every(5).minutes.do(resume_pending_tasks)

    api_app = FastAPI()
    api_app.include_router(app_router)

    HOST = os.environ.get("BACKDOOR_HOST", "0.0.0.0")  # noqa: S104
    PORT = int(os.environ.get("BACKDOOR_PORT", 8000))

    @api_app.get("/", name="Health Check")
    async def read_root() -> JSONResponse:
        """Implement API for health check."""
        return JSONResponse(
            {
                "status": "API is healthy",
                "version": __version__,
            },
            status_code=200,
        )

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    scheduler_thread = threading.Thread(
        target=scheduler_job,
        daemon=True,
    )
    scheduler_thread.start()

    config = uvicorn.Config(
        api_app,
        loop=event_loop,
        host=HOST,
        port=PORT,
        log_level="info",
        timeout_keep_alive=300,
        log_config=logging_config,
        reload=True,
    )

    server = uvicorn.Server(config)
    event_loop.run_until_complete(server.serve())
