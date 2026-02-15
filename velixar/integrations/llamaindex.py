"""LlamaIndex integration for Velixar memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from llama_index.core.memory import BaseMemory as LlamaBaseMemory
    from llama_index.core.bridge.pydantic import Field
except ImportError:
    raise ImportError("Install llama-index: pip install velixar[llamaindex]")

from velixar import Velixar, MemoryTier


class VelixarMemory(LlamaBaseMemory):
    """LlamaIndex memory backed by Velixar.
    
    Usage:
        from velixar.integrations.llamaindex import VelixarMemory
        from llama_index.core.agent import ReActAgent
        
        memory = VelixarMemory(api_key="vlx_...")
        agent = ReActAgent.from_tools(tools, memory=memory)
    """

    client: Velixar = Field(default=None, exclude=True)
    user_id: Optional[str] = None
    max_memories: int = 10

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = Velixar(api_key=api_key)
        self.user_id = user_id

    @classmethod
    def class_name(cls) -> str:
        return "VelixarMemory"

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Get relevant memories for input."""
        if not input:
            return []

        results = self.client.search(
            query=input,
            user_id=self.user_id,
            limit=self.max_memories,
        )

        return [
            {
                "content": mem.content,
                "role": mem.metadata.get("role", "system"),
                "score": mem.score,
            }
            for mem in results.memories
        ]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories (returns recent)."""
        results = self.client.search(
            query="*",
            user_id=self.user_id,
            limit=self.max_memories,
        )
        return [{"content": m.content} for m in results.memories]

    def put(self, message: Dict[str, Any]) -> None:
        """Store a message as memory."""
        content = message.get("content", "")
        role = message.get("role", "user")
        
        if content:
            self.client.store(
                content=content,
                user_id=self.user_id,
                tier=MemoryTier.SESSION,
                metadata={"role": role},
            )

    def set(self, messages: List[Dict[str, Any]]) -> None:
        """Set multiple messages."""
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """Reset is not supported - memories persist."""
        pass
