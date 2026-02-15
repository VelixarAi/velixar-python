"""Velixar Python SDK - Persistent memory for AI applications."""

from velixar.client import Velixar, AsyncVelixar
from velixar.types import Memory, SearchResult, MemoryTier
from velixar.exceptions import (
    VelixarError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
)

__version__ = "0.1.0"
__all__ = [
    "Velixar",
    "AsyncVelixar",
    "Memory",
    "SearchResult",
    "MemoryTier",
    "VelixarError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
]
