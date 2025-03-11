import json
import logging
import os
from enum import Enum

import requests
import schedule
from pydantic import BaseModel
from redis import Redis

from .redis_kit import distributed_scheduling_job, reusable_redis_connection

logger = logging.getLogger(__name__)

TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
TELEGRAM_MESSAGE_LENGTH_LIMIT = 4096
TELEGRAM_MESSAGE_LIST_REDIS_KEY = "telegram_message_list"


class TelegramMessageParseMode(str, Enum):
    """Implement of a Telegram message parse mode."""

    Markdown = "Markdown"
    MarkdownV2 = "MarkdownV2"
    HTML = "HTML"


class TelegramMessage(BaseModel):
    """Model for representing a Telegram message."""

    text: str
    parse_mode: TelegramMessageParseMode = TelegramMessageParseMode.MarkdownV2
    disable_notification: bool = True
    link_preview_options: dict = {}
    room: str

    can_batch: bool = False


def escape_str(s: str) -> str:
    """Escape special characters in a string."""
    rules = [("_", "\\_"), ("*", "\\*"), ("[", "\\["), ("`", "\\`")]

    special = "[#*#]"

    for a, b in rules:
        s = s.replace(b, special)
        s = s.replace(a, b)
        s = s.replace(special, b)

    return s


def get_url(room: str, api_key: str = TELEGRAM_API_KEY) -> str:
    """Get the URL to send a message to a Telegram room."""
    return f"https://api.telegram.org/bot{api_key}/sendMessage?chat_id={room}"


def send_message(
    message_to_send: str,
    room: int,
    preview_opt: dict = {},  # noqa: B006
    fmt: str = TelegramMessageParseMode.MarkdownV2,
    schedule: bool = False,
) -> bool:
    """Send a message to a Telegram room."""
    if fmt == TelegramMessageParseMode.Markdown:
        message_to_send = escape_str(message_to_send)

    if schedule and room is not None:
        _enqueue(
            TelegramMessage(
                text=message_to_send,
                parse_mode=fmt,
                disable_notification=True,
                link_preview_options=preview_opt,
                room=room,
                can_batch=True,
            ),
        )

        return True

    url = get_url(room=room)

    logger.info(
        f"Sending a message of length {len(message_to_send)} to room {room}",
    )
    payload = {
        "text": message_to_send,
        "parse_mode": fmt,
        "disable_notification": True,
        "link_preview_options": json.dumps(preview_opt),
    }

    resp = requests.post(url, json=payload, timeout=5)

    if resp.status_code == 200:
        return True

    logger.error(f"Failed to send message to Telegram: {resp.text}")
    return False


def _enqueue(msg: TelegramMessage) -> None:
    redis_connection: Redis = reusable_redis_connection()
    redis_connection.rpush(
        TELEGRAM_MESSAGE_LIST_REDIS_KEY, json.dumps(msg.model_dump()),
    )


def group_message(
    msgs: list[TelegramMessage],
    separator: str,
    limit_chars: int = TELEGRAM_MESSAGE_LENGTH_LIMIT,
) -> list[list[TelegramMessage]]:
    """Group messages into a single message.

    Messages are grouped if the total length of the messages is less
    than the limit.

    """
    total_length = 0
    grouped_msgs = []
    current_group = []

    for msg in msgs:
        need_separator = len(current_group) > 0
        l_separator = len(separator) if need_separator else 0

        if total_length + len(msg.text) + l_separator > limit_chars:
            grouped_msgs.append(current_group)
            current_group = []
            total_length = 0

        current_group.append(msg)
        total_length += len(msg.text) + l_separator

    if len(current_group) > 0:
        grouped_msgs.append(current_group)

    return grouped_msgs


@distributed_scheduling_job(interval_seconds=20)
def _flush() -> None:
    redis_connection: Redis = reusable_redis_connection()

    length_queue = redis_connection.llen(TELEGRAM_MESSAGE_LIST_REDIS_KEY)
    by_room = {}

    for msg in range(length_queue):
        msg = redis_connection.lpop(TELEGRAM_MESSAGE_LIST_REDIS_KEY)
        msg = json.loads(msg)
        msg = TelegramMessage.model_validate(msg)

        if not msg.can_batch:
            send_message(
                msg.text,
                room=msg.room,
                fmt=msg.parse_mode,
                preview_opt=msg.link_preview_options,
            )

            continue

        if msg.room not in by_room:
            by_room[msg.room] = []

        by_room[msg.room].append(msg)

    sep = "\n" + "-" * 10 + "\n"

    for room, msgs in by_room.items():
        groups = group_message(msgs, sep)

        for group in groups:
            joint_message = sep.join([msg.text for msg in group])
            send_message(
                joint_message,
                room=room,
                fmt=TelegramMessageParseMode.HTML,
            )


schedule.every(20).seconds.do(_flush)
