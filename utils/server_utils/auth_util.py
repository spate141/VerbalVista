from typing import List
from fastapi import Depends, status, HTTPException
from fastapi.security.api_key import APIKeyHeader


class AuthUtil:

    def __init__(self, valid_api_keys: List[str] = None):
        """
        Initializes the AuthUtil instance with a list of valid API keys.

        :param valid_api_keys: A list of strings representing valid API keys that can access the service.
        """

        self.valid_api_keys = valid_api_keys

    async def get_api_key(self, api_key_header: str = Depends(APIKeyHeader(name="X-API-Key"))) -> str:
        """
        Validates the provided API key against the list of valid API keys. This method is intended to be
        used as a dependency in FastAPI route handlers to enforce API key authentication.

        :param api_key_header: The API key extracted from the request headers.
        :return: The API key if it is valid.
        :raises HTTPException: If the API key is not valid, an HTTP 403 Forbidden error is raised.
        """
        if api_key_header not in self.valid_api_keys:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
        return api_key_header

