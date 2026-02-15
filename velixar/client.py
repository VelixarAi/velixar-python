"""Velixar client implementations."""

from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Optional, Sequence
from contextlib import contextmanager, asynccontextmanager

import httpx

from velixar.types import (
    Memory, MemoryTier, SearchResult,
    StoreRequest, StoreResponse, BatchStoreResponse,
)
from velixar.exceptions import (
    VelixarError, AuthenticationError, RateLimitError,
    NotFoundError, ValidationError, InsufficientScopeError,
)


DEFAULT_BASE_URL = "https://api.velixarai.com/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


class BaseClient:
    """Base client with shared configuration."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("VELIXAR_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set VELIXAR_API_KEY or pass api_key=")
        
        self.base_url = (base_url or os.environ.get("VELIXAR_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "velixar-python/0.1.0",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Convert HTTP errors to SDK exceptions."""
        if response.status_code == 401:
            raise AuthenticationError()
        elif response.status_code == 403:
            raise InsufficientScopeError()
        elif response.status_code == 404:
            raise NotFoundError()
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(retry_after=int(retry_after) if retry_after else None)
        elif response.status_code == 400:
            data = response.json()
            raise ValidationError(data.get("error", "Invalid request"))
        elif response.status_code >= 500:
            raise VelixarError(f"Server error: {response.status_code}", response.status_code)


class Velixar(BaseClient):
    """Synchronous Velixar client."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Velixar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic."""
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.request(method, path, json=json, params=params)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    if attempt < self.max_retries - 1:
                        time.sleep(retry_after)
                        continue
                
                if response.status_code >= 400:
                    self._handle_error(response)
                
                return response.json()
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise VelixarError(f"Connection failed: {e}")
        
        raise last_error or VelixarError("Request failed")

    # ========== Memory Operations ==========

    def store(
        self,
        content: str,
        *,
        user_id: str | None = None,
        tier: MemoryTier | int = MemoryTier.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory. Returns memory ID."""
        data = {
            "content": content,
            "tier": int(tier),
        }
        if user_id:
            data["user_id"] = user_id
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata

        result = self._request("POST", "/memory", json=data)
        return result["id"]

    def store_many(self, memories: Sequence[dict[str, Any] | StoreRequest]) -> BatchStoreResponse:
        """Store multiple memories in one request."""
        items = []
        for m in memories:
            if isinstance(m, StoreRequest):
                items.append(m.model_dump())
            else:
                items.append(m)
        
        result = self._request("POST", "/memory/batch", json={"memories": items})
        return BatchStoreResponse(**result)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: str | None = None,
        tiers: list[MemoryTier | int] | None = None,
    ) -> SearchResult:
        """Search memories by semantic similarity."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if user_id:
            params["user_id"] = user_id
        if tiers:
            params["tiers"] = ",".join(str(int(t)) for t in tiers)

        result = self._request("GET", "/memory/search", params=params)
        return SearchResult(
            memories=[Memory(**m) for m in result.get("memories", [])],
            count=result.get("count", 0),
            query=query,
        )

    def get(self, memory_id: str) -> Memory:
        """Get a specific memory by ID."""
        result = self._request("GET", f"/memory/{memory_id}")
        return Memory(**result.get("memory", result))

    def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted."""
        result = self._request("DELETE", f"/memory/{memory_id}")
        return result.get("deleted", False)

    def get_context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """Get formatted context string for LLM prompts."""
        results = self.search(query, limit=10, user_id=user_id)
        
        context_parts = []
        token_estimate = 0
        
        for mem in results.memories:
            mem_tokens = len(mem.content.split()) * 1.3
            if token_estimate + mem_tokens > max_tokens:
                break
            context_parts.append(mem.content)
            token_estimate += mem_tokens
        
        return "\n\n".join(context_parts)


class AsyncVelixar(BaseClient):
    """Asynchronous Velixar client."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> AsyncVelixar:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make async HTTP request with retry logic."""
        last_error: Exception | None = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, path, json=json, params=params)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                
                if response.status_code >= 400:
                    self._handle_error(response)
                
                return response.json()
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise VelixarError(f"Connection failed: {e}")
        
        raise last_error or VelixarError("Request failed")

    async def store(
        self,
        content: str,
        *,
        user_id: str | None = None,
        tier: MemoryTier | int = MemoryTier.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        data = {"content": content, "tier": int(tier)}
        if user_id:
            data["user_id"] = user_id
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        result = await self._request("POST", "/memory", json=data)
        return result["id"]

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        user_id: str | None = None,
        tiers: list[MemoryTier | int] | None = None,
    ) -> SearchResult:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if user_id:
            params["user_id"] = user_id
        if tiers:
            params["tiers"] = ",".join(str(int(t)) for t in tiers)
        result = await self._request("GET", "/memory/search", params=params)
        return SearchResult(
            memories=[Memory(**m) for m in result.get("memories", [])],
            count=result.get("count", 0),
            query=query,
        )

    async def get(self, memory_id: str) -> Memory:
        result = await self._request("GET", f"/memory/{memory_id}")
        return Memory(**result.get("memory", result))

    async def delete(self, memory_id: str) -> bool:
        result = await self._request("DELETE", f"/memory/{memory_id}")
        return result.get("deleted", False)
