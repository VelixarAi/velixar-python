"""LangChain integration for Velixar memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, messages_from_dict
except ImportError:
    raise ImportError("Install langchain: pip install velixar[langchain]")

from velixar import Velixar, MemoryTier


class VelixarChatMessageHistory(BaseChatMessageHistory):
    """Velixar-backed chat message history for RunnableWithMessageHistory.

    This is the recommended integration. Use with any LangChain runnable:

        from langchain_core.runnables.history import RunnableWithMessageHistory
        from velixar.integrations.langchain import VelixarChatMessageHistory

        def get_session_history(session_id: str) -> VelixarChatMessageHistory:
            return VelixarChatMessageHistory(session_id=session_id, api_key="vlx_...")

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    """

    def __init__(
        self,
        session_id: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        tier: MemoryTier = MemoryTier.SESSION,
        max_messages: int = 50,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.tier = tier
        self.max_messages = max_messages
        self.client = Velixar(api_key=api_key)

    @property
    def messages(self) -> List[BaseMessage]:
        results = self.client.search(
            query=f"session:{self.session_id}",
            user_id=self.user_id,
            limit=self.max_messages,
        )
        msgs: List[BaseMessage] = []
        for mem in results.memories:
            if mem.metadata.get("session_id") != self.session_id:
                continue
            if mem.metadata.get("role") == "human":
                msgs.append(HumanMessage(content=mem.content))
            else:
                msgs.append(AIMessage(content=mem.content))
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        role = "human" if isinstance(message, HumanMessage) else "ai"
        self.client.store(
            content=message.content,
            user_id=self.user_id,
            tier=self.tier,
            tags=["langchain", self.session_id],
            metadata={"role": role, "session_id": self.session_id},
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        pass  # Velixar memories persist by design


class VelixarSemanticMemory:
    """Semantic long-term memory that injects relevant context into prompts.

    Unlike chat history (which replays messages), this searches all stored
    knowledge and returns the most relevant memories for the current input.

    Use as a retriever-style component in your chain:

        from velixar.integrations.langchain import VelixarSemanticMemory

        mem = VelixarSemanticMemory(api_key="vlx_...")

        # In your chain, inject context before the LLM call
        context = mem.get_context("What does the user prefer?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        max_memories: int = 10,
        max_tokens: int = 2000,
    ):
        self.client = Velixar(api_key=api_key)
        self.user_id = user_id
        self.max_memories = max_memories
        self.max_tokens = max_tokens

    def get_context(self, query: str) -> str:
        return self.client.get_context(
            query, user_id=self.user_id, max_tokens=self.max_tokens
        )

    def store(self, content: str, tier: MemoryTier = MemoryTier.SEMANTIC, **kwargs: Any) -> str:
        return self.client.store(
            content=content, user_id=self.user_id, tier=tier, **kwargs
        )


# ---------------------------------------------------------------------------
# Legacy support: VelixarMemory for ConversationChain users
# ConversationChain is deprecated but still widely used
# ---------------------------------------------------------------------------

_has_base_memory = False
try:
    from langchain_core.memory import BaseMemory
    _has_base_memory = True
except ImportError:
    pass

if _has_base_memory:

    class VelixarMemory(BaseMemory):  # type: ignore[misc]
        """Legacy LangChain BaseMemory for ConversationChain.

        Note: ConversationChain and BaseMemory are deprecated in LangChain >=0.3.
        Prefer VelixarChatMessageHistory with RunnableWithMessageHistory.
        """

        client: Any = None
        user_id: Optional[str] = None
        memory_key: str = "history"
        input_key: str = "input"
        output_key: str = "output"
        return_messages: bool = False
        max_context_memories: int = 10
        auto_store: bool = True
        store_tier: int = MemoryTier.SESSION

        def __init__(self, api_key: Optional[str] = None, user_id: Optional[str] = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.client = Velixar(api_key=api_key)
            self.user_id = user_id

        @property
        def memory_variables(self) -> List[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs.get(self.input_key, "")
            if not query:
                return {self.memory_key: [] if self.return_messages else ""}
            results = self.client.search(query=query, user_id=self.user_id, limit=self.max_context_memories)
            if self.return_messages:
                msgs: List[BaseMessage] = []
                for mem in results.memories:
                    if mem.metadata.get("role") == "ai":
                        msgs.append(AIMessage(content=mem.content))
                    else:
                        msgs.append(HumanMessage(content=mem.content))
                return {self.memory_key: msgs}
            return {self.memory_key: "\n".join(m.content for m in results.memories)}

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            if not self.auto_store:
                return
            human_input = inputs.get(self.input_key, "")
            ai_output = outputs.get(self.output_key, "")
            if human_input:
                self.client.store(content=human_input, user_id=self.user_id, tier=self.store_tier, metadata={"role": "human"})
            if ai_output:
                self.client.store(content=ai_output, user_id=self.user_id, tier=self.store_tier, metadata={"role": "ai"})

        def clear(self) -> None:
            pass
