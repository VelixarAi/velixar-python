"""LangChain agent with Velixar memory."""

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from velixar.integrations.langchain import VelixarMemory, VelixarChatMessageHistory

# ============================================================
# SIMPLE CONVERSATION CHAIN
# ============================================================

def simple_conversation():
    """Basic conversation with persistent memory."""
    
    memory = VelixarMemory(
        api_key="vlx_your_key",
        user_id="user_123",
        auto_store=True,  # Automatically store conversation turns
    )
    
    chain = ConversationChain(
        llm=ChatOpenAI(model="gpt-4"),
        memory=memory,
    )
    
    # First conversation
    print(chain.invoke({"input": "My name is Alice and I love hiking"}))
    print(chain.invoke({"input": "I also enjoy photography"}))
    
    # Later conversation (memory persists!)
    print(chain.invoke({"input": "What do you know about me?"}))


# ============================================================
# AGENT WITH TOOLS
# ============================================================

def agent_with_memory():
    """Agent that uses Velixar for long-term memory."""
    
    from langchain_core.tools import tool
    
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"
    
    @tool
    def search_web(query: str) -> str:
        """Search the web."""
        return f"Search results for '{query}': ..."
    
    memory = VelixarMemory(
        api_key="vlx_your_key",
        user_id="user_123",
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with persistent memory. "
                   "Remember important facts about the user."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4")
    tools = [get_weather, search_web]
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
    
    # Use the agent
    print(executor.invoke({"input": "I live in Seattle. What's the weather like?"}))
    print(executor.invoke({"input": "Where do I live?"}))  # Uses memory


# ============================================================
# CHAT MESSAGE HISTORY (for RunnableWithMessageHistory)
# ============================================================

def with_message_history():
    """Use VelixarChatMessageHistory for session-based memory."""
    
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    def get_session_history(session_id: str):
        return VelixarChatMessageHistory(
            api_key="vlx_your_key",
            user_id="user_123",
            session_id=session_id,
        )
    
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # Use with session
    config = {"configurable": {"session_id": "session_abc123"}}
    
    print(with_history.invoke({"input": "Hi, I'm Bob"}, config=config))
    print(with_history.invoke({"input": "What's my name?"}, config=config))


if __name__ == "__main__":
    simple_conversation()
