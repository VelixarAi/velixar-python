"""Type definitions for Velixar SDK."""

from datetime import datetime
from enum import IntEnum
from typing import Any, Optional
from pydantic import BaseModel, Field


class MemoryTier(IntEnum):
    """Memory storage tiers."""
    PINNED = 0      # Critical facts, never expire
    SESSION = 1     # Current session context
    SEMANTIC = 2    # Long-term semantic memories
    ORG = 3         # Organization-wide knowledge


class Memory(BaseModel):
    """A stored memory."""
    id: str
    content: str
    tier: MemoryTier = MemoryTier.SEMANTIC
    user_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None  # Relevance score from search
    created_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class SearchResult(BaseModel):
    """Search results container."""
    memories: list[Memory]
    count: int
    query: str


class StoreRequest(BaseModel):
    """Request to store a memory."""
    content: str
    user_id: Optional[str] = None
    tier: MemoryTier = MemoryTier.SEMANTIC
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    """Response from storing a memory."""
    id: str
    stored: bool


class BatchStoreRequest(BaseModel):
    """Request to store multiple memories."""
    memories: list[StoreRequest]


class BatchStoreResponse(BaseModel):
    """Response from batch store."""
    ids: list[str]
    stored: int
    failed: int = 0
