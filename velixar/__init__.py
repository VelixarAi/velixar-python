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

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:  # derived, never hand-typed — a literal here drifted to 1.0.0 in a 1.0.1 wheel
    __version__ = _pkg_version("velixar")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"
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
