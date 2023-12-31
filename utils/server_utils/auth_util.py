from typing import List
from fastapi import Depends, status, HTTPException
from fastapi.security.api_key import APIKeyHeader


class AuthUtil:

    def __init__(self, valid_api_keys: List[str] = None):
        self.valid_api_keys = valid_api_keys

    async def get_api_key(self, api_key_header: str = Depends(APIKeyHeader(name="X-API-Key"))):
        """
        Asynchronous method to validate the provided API key against a list of valid API keys.

        :param api_key_header: The API key obtained from the request header 'X-API-Key'.
        :type api_key_header: str
        :return: The validated API key if it is found in the list of valid API keys.
        :rtype: str
        :raises HTTPException: If the provided API key is not found in the list of valid API keys,
                               an HTTP exception with status code 403 (Forbidden) is raised.
        """
        if api_key_header not in self.valid_api_keys:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
        return api_key_header

