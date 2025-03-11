from collections.abc import Callable
from hashlib import sha256
from typing import Annotated, TypeVar

from fastapi import Header, HTTPException

from . import constants as const

SECRET_TOKEN = const.API_SECRET_TOKEN
sha256_of_secret_token = sha256(SECRET_TOKEN.encode()).hexdigest()

TokenType = TypeVar("TokenType", Annotated[str | None, Header()])


async def verify_opencall_x_token(x_token: TokenType = None) -> TokenType:
    """Verify opencall x-token."""
    global sha256_of_secret_token

    if x_token != sha256_of_secret_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return x_token


async def verify_x_token(x_token: TokenType = None) -> TokenType:
    """Verify x-token."""
    if x_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="X-Token header invalid")

    return x_token


def verify_third_party_authorization_key(
    backend_url: str,
    headers: dict = {},  # noqa: B006
) -> Callable:
    """Verify third party authorization key."""
    async def wrapper(
        authorization: TokenType = "",
    ) -> bool:
        if authorization != "":
            return True

        raise HTTPException(status_code=401, detail="Unauthorized")

    return wrapper
