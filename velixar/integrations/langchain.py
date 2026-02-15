"""LangChain integration for Velixar memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
except ImportError:
    raise ImportError("Install langchain: pip install velixar[langchain]")

from velixar import Velixar, AsyncVelixar, MemoryTier


class VelixarMemory(BaseMemory):
    """LangChain memory backed by Velixar.
    
    Usage:
        from velixar.integrations.langchain import VelixarMemory
        from langchain.chains import ConversationChain
        
        memory = VelixarMemory(api_key="vlx_...")
        chain = ConversationChain(llm=llm, memory=memory)
    """

    client: Velixar = None  # type: ignore
    user_id: Optional[str] = None
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False
    max_context_memories: int = 10
    auto_store: bool = True
    store_tier: MemoryTier = MemoryTier.SESSION

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.client = Velixar(api_key=api_key)
        self.user_id = user_id

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories for the current input."""
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: [] if self.return_messages else ""}

        results = self.client.search(
            query=query,
            user_id=self.user_id,
            limit=self.max_context_memories,
        )

        if self.return_messages:
            messages: List[BaseMessage] = []
            for mem in results.memories:
                # Reconstruct as conversation turns if possible
                if mem.metadata.get("role") == "human":
                    messages.append(HumanMessage(content=mem.content))
                elif mem.metadata.get("role") == "ai":
                    messages.append(AIMessage(content=mem.content))
                else:
                    # Generic context
                    messages.append(HumanMessage(content=f"[Context] {mem.content}"))
            return {self.memory_key: messages}
        else:
            context = "\n".join(m.content for m in results.memories)
            return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation turn to Velixar."""
        if not self.auto_store:
            return

        human_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if human_input:
            self.client.store(
                content=human_input,
                user_id=self.user_id,
                tier=self.store_tier,
                metadata={"role": "human"},
            )

        if ai_output:
            self.client.store(
                content=ai_output,
                user_id=self.user_id,
                tier=self.store_tier,
                metadata={"role": "ai"},
            )

    def clear(self) -> None:
        """Clear is not supported - memories persist."""
        pass


class VelixarChatMessageHistory:
    """Chat message history backed by Velixar.
    
    For use with RunnableWithMessageHistory.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.client = Velixar(api_key=api_key)
        self.user_id = user_id
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages for this session."""
        results = self.client.search(
            query="conversation history",
            user_id=self.user_id,
            limit=50,
            tiers=[MemoryTier.SESSION],
        )
        
        messages: List[BaseMessage] = []
        for mem in results.memories:
            if mem.metadata.get("session_id") == self.session_id:
                if mem.metadata.get("role") == "human":
                    messages.append(HumanMessage(content=mem.content))
                else:
                    messages.append(AIMessage(content=mem.content))
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to history."""
        role = "human" if isinstance(message, HumanMessage) else "ai"
        self.client.store(
            content=message.content,
            user_id=self.user_id,
            tier=MemoryTier.SESSION,
            metadata={"role": role, "session_id": self.session_id},
        )

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        pass  # Memories persist
