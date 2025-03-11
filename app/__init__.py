__version__ = "v3.2.7"

import logging
from pathlib import Path

from dotenv import load_dotenv

from . import wrappers

logger = logging.getLogger(__name__)
if not load_dotenv(Path(__file__).parent.parent / ".env"):
    logger.warning("No .env file found")
