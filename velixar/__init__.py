"""Velixar Python SDK v1.0.0 — Persistent cognitive memory for AI applications."""

from velixar.client import Velixar, AsyncVelixar, GraphEntity, GraphRelation, TraverseResult
from velixar.types import Memory, SearchResult, MemoryTier
from velixar.exceptions import (
    VelixarError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
)

__version__ = "1.0.0"
__all__ = [
    "Velixar",
    "AsyncVelixar",
    "Memory",
    "SearchResult",
    "MemoryTier",
    "GraphEntity",
    "GraphRelation",
    "TraverseResult",
    "VelixarError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
]
