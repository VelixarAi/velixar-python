# Velixar Python SDK

[![PyPI](https://img.shields.io/pypi/v/velixar)](https://pypi.org/project/velixar/)
[![Python](https://img.shields.io/pypi/pyversions/velixar)](https://pypi.org/project/velixar/)
[![License](https://img.shields.io/github/license/VelixarAi/velixar-python)](LICENSE)

Persistent memory for AI assistants and agents. Give any LLM-powered application long-term recall across sessions.

Velixar is an open memory layer — it works with any AI assistant, agent framework, or LLM pipeline. Store facts, preferences, and context that persist beyond a single conversation.

## Installation

```bash
pip install velixar

# With LangChain integration
pip install velixar[langchain]

# With LlamaIndex integration
pip install velixar[llamaindex]

# All integrations
pip install velixar[all]
```

## Quick Start

```python
from velixar import Velixar

v = Velixar(api_key="vlx_your_key")  # Or set VELIXAR_API_KEY env var

# Store a memory
memory_id = v.store(
    content="User prefers dark mode and metric units",
    tier=0,  # 0=pinned, 1=session, 2=semantic (default), 3=org
    user_id="user_123",
    tags=["preferences"],
)

# Search memories semantically
results = v.search("user preferences", limit=5)
for memory in results.memories:
    print(f"[{memory.score:.2f}] {memory.content}")

# Get context for LLM prompts
context = v.get_context("What are the user's preferences?", max_tokens=2000)
```

## Async Support

```python
from velixar import AsyncVelixar

async with AsyncVelixar(api_key="vlx_...") as v:
    await v.store("User's favorite color is blue", user_id="user_123")
    results = await v.search("favorite color")
```

## Memory Tiers

| Tier | Name | Use Case |
|------|------|----------|
| 0 | Pinned | Critical facts, user preferences, never expire |
| 1 | Session | Current conversation context |
| 2 | Semantic | Long-term memories (default) |
| 3 | Organization | Shared team knowledge |

```python
from velixar import MemoryTier

v.store("User is allergic to peanuts", tier=MemoryTier.PINNED)
v.store("Currently discussing project X", tier=MemoryTier.SESSION)
```

## Use With Any AI Assistant

Velixar is assistant-agnostic. Plug it into OpenAI, Anthropic, LangChain, LlamaIndex, custom agents, or any LLM pipeline:

```python
# Inject memories as context before calling your LLM
results = v.search(user_message, limit=5)
context = "\n".join(m.content for m in results.memories)

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"Relevant memories:\n{context}"},
        {"role": "user", "content": user_message},
    ],
)

# Store important facts after the conversation
v.store("User prefers concise answers", user_id="user_123")
```

## LangChain Integration

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from velixar.integrations.langchain import VelixarMemory

memory = VelixarMemory(api_key="vlx_...", user_id="user_123")

chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
chain.invoke({"input": "Remember that I prefer Python over JavaScript"})
chain.invoke({"input": "What programming language do I prefer?"})
```

## LlamaIndex Integration

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from velixar.integrations.llamaindex import VelixarMemory

memory = VelixarMemory(api_key="vlx_...", user_id="user_123")
agent = ReActAgent.from_tools(tools=[...], llm=OpenAI(), memory=memory)
```

## Batch Operations

```python
result = v.store_many([
    {"content": "Fact 1", "tier": 0},
    {"content": "Fact 2", "tier": 2, "tags": ["important"]},
    {"content": "Fact 3", "user_id": "user_456"},
])
```

## Error Handling

```python
from velixar import VelixarError, RateLimitError, AuthenticationError

try:
    v.store("test")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except VelixarError as e:
    print(f"Error: {e.message}")
```

## Configuration

```python
v = Velixar(
    api_key="vlx_...",           # Or VELIXAR_API_KEY env var
    base_url="https://...",      # Custom endpoint (optional)
    timeout=30.0,                # Request timeout in seconds
    max_retries=3,               # Retry attempts for failures
)
```

## Get an API Key

Sign up at [velixarai.com](https://velixarai.com) and generate a key under Settings → API Keys.

## Related

- [velixar (JavaScript SDK)](https://github.com/VelixarAi/velixar-js) — TypeScript/JavaScript client
- [velixar-mcp-server](https://github.com/VelixarAi/velixar-mcp-server) — MCP server for any MCP-compatible AI assistant

## License

MIT
