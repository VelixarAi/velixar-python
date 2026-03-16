"""Velixar Python SDK v1.0.0 — Persistent cognitive memory for AI applications."""

from __future__ import annotations

import os
import time
import asyncio
from typing import Any, Sequence
from dataclasses import dataclass, field

import httpx

from velixar.types import Memory, MemoryTier, SearchResult, StoreRequest, BatchStoreResponse
from velixar.exceptions import (
    VelixarError, AuthenticationError, RateLimitError,
    NotFoundError, ValidationError, InsufficientScopeError,
)

DEFAULT_BASE_URL = "https://api.velixarai.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class GraphEntity:
    id: str
    entity_type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    relevance: float | None = None


@dataclass
class GraphRelation:
    source: str
    target: str
    relation_type: str
    weight: float | None = None


@dataclass
class TraverseResult:
    nodes: list[GraphEntity]
    edges: list[GraphRelation]


class BaseClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        workspace_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or os.environ.get("VELIXAR_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set VELIXAR_API_KEY or pass api_key=")
        self.base_url = (base_url or os.environ.get("VELIXAR_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.workspace_id = workspace_id or os.environ.get("VELIXAR_WORKSPACE_ID")
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "velixar-python/1.0.0",
        }
        if self.workspace_id:
            h["X-Workspace-Id"] = self.workspace_id
        return h

    def _handle_error(self, response: httpx.Response) -> None:
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
    """Synchronous Velixar client with full API coverage."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, headers=self._headers, timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Velixar:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    # ── Memory CRUD ──

    def store(self, content: str, *, user_id: str | None = None, tier: MemoryTier | int = MemoryTier.SEMANTIC, tags: list[str] | None = None) -> str:
        data: dict[str, Any] = {"content": content, "tier": int(tier)}
        if user_id: data["user_id"] = user_id
        if tags: data["tags"] = tags
        return self._request("POST", "/memory", json=data)["id"]

    def store_many(self, memories: Sequence[dict[str, Any] | StoreRequest]) -> BatchStoreResponse:
        items = [m.model_dump() if isinstance(m, StoreRequest) else m for m in memories]
        return BatchStoreResponse(**self._request("POST", "/memory/batch", json={"memories": items}))

    def search(self, query: str, *, limit: int = 10, user_id: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if user_id: params["user_id"] = user_id
        result = self._request("GET", "/memory/search", params=params)
        return SearchResult(memories=[Memory(**m) for m in result.get("memories", [])], count=result.get("count", 0), query=query)

    def get(self, memory_id: str) -> Memory:
        result = self._request("GET", f"/memory/{memory_id}")
        return Memory(**result.get("memory", result))

    def update(self, memory_id: str, *, content: str | None = None, tags: list[str] | None = None) -> bool:
        data: dict[str, Any] = {}
        if content is not None: data["content"] = content
        if tags is not None: data["tags"] = tags
        return self._request("PATCH", f"/memory/{memory_id}", json=data).get("updated", False)

    def list(self, *, limit: int = 10, cursor: str | None = None, user_id: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"limit": limit}
        if cursor: params["cursor"] = cursor
        if user_id: params["user_id"] = user_id
        result = self._request("GET", "/memory/list", params=params)
        return SearchResult(memories=[Memory(**m) for m in result.get("memories", [])], count=result.get("count", 0), query="", cursor=result.get("cursor"))

    def delete(self, memory_id: str) -> bool:
        return self._request("DELETE", f"/memory/{memory_id}").get("deleted", False)

    # ── Graph ──

    def graph_traverse(self, entity: str, *, depth: int = 2) -> TraverseResult:
        result = self._request("POST", "/graph/traverse", json={"entity": entity, "max_hops": min(depth, 10)})
        return TraverseResult(
            nodes=[GraphEntity(**n) for n in result.get("nodes", [])],
            edges=[GraphRelation(**e) for e in result.get("edges", [])],
        )

    def graph_search(self, query: str, *, entity_type: str | None = None, limit: int = 20) -> list[GraphEntity]:
        result = self._request("POST", "/graph/search", json={"query": query, "entity_type": entity_type, "limit": limit})
        return [GraphEntity(**e) for e in result.get("entities", result.get("results", []))]

    def graph_entities(self, *, limit: int = 20) -> list[GraphEntity]:
        result = self._request("GET", "/graph/entities", params={"limit": limit})
        return [GraphEntity(**e) for e in result.get("entities", [])]

    # ── Identity ──

    def get_identity(self, *, user_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if user_id: params["user_id"] = user_id
        return self._request("GET", "/memory/identity", params=params)

    # ── Exocortex ──

    def overview(self) -> dict[str, Any]:
        return self._request("GET", "/exocortex/overview")

    def contradictions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/exocortex/contradictions").get("contradictions", [])

    # ── CI/CD Webhook ──

    def webhook(self, event_type: str, content: str, *, tags: list[str] | None = None, **metadata: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"event_type": event_type, "content": content}
        if tags: body["tags"] = tags
        body.update(metadata)
        return self._request("POST", "/webhook/ci", json=body)

    # ── Health ──

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")


class AsyncVelixar(BaseClient):
    """Asynchronous Velixar client with full API coverage."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> AsyncVelixar:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(self, method: str, path: str, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
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

    # ── Memory CRUD ──

    async def store(self, content: str, *, user_id: str | None = None, tier: MemoryTier | int = MemoryTier.SEMANTIC, tags: list[str] | None = None) -> str:
        data: dict[str, Any] = {"content": content, "tier": int(tier)}
        if user_id: data["user_id"] = user_id
        if tags: data["tags"] = tags
        return (await self._request("POST", "/memory", json=data))["id"]

    async def search(self, query: str, *, limit: int = 10, user_id: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if user_id: params["user_id"] = user_id
        result = await self._request("GET", "/memory/search", params=params)
        return SearchResult(memories=[Memory(**m) for m in result.get("memories", [])], count=result.get("count", 0), query=query)

    async def get(self, memory_id: str) -> Memory:
        result = await self._request("GET", f"/memory/{memory_id}")
        return Memory(**result.get("memory", result))

    async def update(self, memory_id: str, *, content: str | None = None, tags: list[str] | None = None) -> bool:
        data: dict[str, Any] = {}
        if content is not None: data["content"] = content
        if tags is not None: data["tags"] = tags
        return (await self._request("PATCH", f"/memory/{memory_id}", json=data)).get("updated", False)

    async def list(self, *, limit: int = 10, cursor: str | None = None, user_id: str | None = None) -> SearchResult:
        params: dict[str, Any] = {"limit": limit}
        if cursor: params["cursor"] = cursor
        if user_id: params["user_id"] = user_id
        result = await self._request("GET", "/memory/list", params=params)
        return SearchResult(memories=[Memory(**m) for m in result.get("memories", [])], count=result.get("count", 0), query="", cursor=result.get("cursor"))

    async def delete(self, memory_id: str) -> bool:
        return (await self._request("DELETE", f"/memory/{memory_id}")).get("deleted", False)

    # ── Graph ──

    async def graph_traverse(self, entity: str, *, depth: int = 2) -> TraverseResult:
        result = await self._request("POST", "/graph/traverse", json={"entity": entity, "max_hops": min(depth, 10)})
        return TraverseResult(
            nodes=[GraphEntity(**n) for n in result.get("nodes", [])],
            edges=[GraphRelation(**e) for e in result.get("edges", [])],
        )

    async def graph_search(self, query: str, *, entity_type: str | None = None, limit: int = 20) -> list[GraphEntity]:
        result = await self._request("POST", "/graph/search", json={"query": query, "entity_type": entity_type, "limit": limit})
        return [GraphEntity(**e) for e in result.get("entities", result.get("results", []))]

    async def graph_entities(self, *, limit: int = 20) -> list[GraphEntity]:
        result = await self._request("GET", "/graph/entities", params={"limit": limit})
        return [GraphEntity(**e) for e in result.get("entities", [])]

    # ── Identity ──

    async def get_identity(self, *, user_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if user_id: params["user_id"] = user_id
        return await self._request("GET", "/memory/identity", params=params)

    # ── Exocortex ──

    async def overview(self) -> dict[str, Any]:
        return await self._request("GET", "/exocortex/overview")

    async def contradictions(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/exocortex/contradictions")).get("contradictions", [])

    # ── CI/CD Webhook ──

    async def webhook(self, event_type: str, content: str, *, tags: list[str] | None = None, **metadata: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"event_type": event_type, "content": content}
        if tags: body["tags"] = tags
        body.update(metadata)
        return await self._request("POST", "/webhook/ci", json=body)

    # ── Health ──

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")
