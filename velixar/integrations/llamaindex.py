"""LlamaIndex integration for Velixar memory."""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from llama_index.core.memory import BaseMemory
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
except ImportError:
    raise ImportError(
        "llama-index-core is required for this integration. "
        "Install it with: pip install velixar[llamaindex]"
    )

from pydantic import PrivateAttr

from velixar import Velixar, MemoryTier


class VelixarMemory(BaseMemory):
    """LlamaIndex memory backed by Velixar persistent storage.

    Stores conversation messages as Velixar memories and retrieves
    semantically relevant context for each new input.

    Usage::

        from velixar.integrations.llamaindex import VelixarMemory
        from llama_index.core.agent import ReActAgent

        memory = VelixarMemory.from_defaults(api_key="vlx_...")
        agent = ReActAgent.from_tools(tools, memory=memory)
    """

    user_id: Optional[str] = None
    max_memories: int = 10

    _client: Velixar = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        max_memories: int = 10,
        **kwargs: Any,
    ):
        super().__init__(user_id=user_id, max_memories=max_memories, **kwargs)
        self._client = Velixar(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "VelixarMemory"

    @classmethod
    def from_defaults(
        cls,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        max_memories: int = 10,
        **kwargs: Any,
    ) -> "VelixarMemory":
        """Create VelixarMemory from defaults."""
        return cls(api_key=api_key, user_id=user_id, max_memories=max_memories, **kwargs)

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        """Retrieve semantically relevant memories as ChatMessages."""
        if not input:
            return []

        results = self._client.search(
            query=input,
            user_id=self.user_id,
            limit=self.max_memories,
        )

        messages = []
        for mem in results.memories:
            role_str = mem.metadata.get("role", "system")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.SYSTEM
            messages.append(ChatMessage(role=role, content=mem.content))
        return messages

    def get_all(self) -> List[ChatMessage]:
        """Get all stored memories as ChatMessages."""
        results = self._client.search(
            query="*",
            user_id=self.user_id,
            limit=self.max_memories,
        )
        return [
            ChatMessage(role=MessageRole.USER, content=m.content)
            for m in results.memories
        ]

    def put(self, message: ChatMessage) -> None:
        """Store a ChatMessage as a Velixar memory."""
        content = message.content
        if content:
            self._client.store(
                content=content,
                user_id=self.user_id,
                tier=MemoryTier.SESSION,
                metadata={"role": message.role.value},
            )

    def set(self, messages: List[ChatMessage]) -> None:
        """Store multiple ChatMessages."""
        for msg in messages:
            self.put(msg)

    def reset(self) -> None:
        """No-op — Velixar memories are persistent by design."""
        pass
