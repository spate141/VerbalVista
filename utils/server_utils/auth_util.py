from typing import List
from fastapi import Depends, status, HTTPException
from fastapi.security.api_key import APIKeyHeader


class AuthUtil:

    def __init__(self, valid_api_keys: List[str] = None):
        self.valid_api_keys = valid_api_keys

    async def get_api_key(self, api_key_header: str = Depends(APIKeyHeader(name="X-API-Key"))):
        if api_key_header not in self.valid_api_keys:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
        return api_key_header

