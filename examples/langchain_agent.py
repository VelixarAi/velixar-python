"""LangChain + Velixar memory examples."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from velixar.integrations.langchain import (
    VelixarChatMessageHistory,
    VelixarSemanticMemory,
    VelixarMemory,
)

API_KEY = "vlx_your_key"


# ============================================================
# 1. RECOMMENDED: RunnableWithMessageHistory
# ============================================================

def modern_chat():
    """Persistent chat history using the current LangChain API."""

    def get_session_history(session_id: str) -> VelixarChatMessageHistory:
        return VelixarChatMessageHistory(session_id=session_id, api_key=API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | ChatOpenAI(model="gpt-4")

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-session-1"}}

    # First conversation
    print(with_history.invoke({"input": "My name is Alice"}, config=config))

    # Later — memory persists across restarts
    print(with_history.invoke({"input": "What's my name?"}, config=config))


# ============================================================
# 2. SEMANTIC MEMORY: inject long-term knowledge into prompts
# ============================================================

def semantic_memory():
    """Use Velixar as a knowledge layer, not just chat replay."""

    mem = VelixarSemanticMemory(api_key=API_KEY, user_id="user_123")

    # Store facts over time
    mem.store("User prefers dark mode and metric units")
    mem.store("User's project deadline is March 15")

    # Later — retrieve relevant context for any query
    context = mem.get_context("What are the user's preferences?")

    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(
        f"Relevant context:\n{context}\n\nUser: Summarize what you know about me."
    )
    print(response)


# ============================================================
# 3. LEGACY: ConversationChain (deprecated but still common)
# ============================================================

def legacy_conversation():
    """For existing code using ConversationChain."""
    from langchain.chains import ConversationChain

    memory = VelixarMemory(api_key=API_KEY, user_id="user_123")
    chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

    print(chain.invoke({"input": "I love hiking and photography"}))
    print(chain.invoke({"input": "What do you know about me?"}))


if __name__ == "__main__":
    modern_chat()
