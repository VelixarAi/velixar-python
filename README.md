# Velixar Python SDK

Persistent memory infrastructure for AI applications.

[![PyPI](https://img.shields.io/pypi/v/velixar)](https://pypi.org/project/velixar/)
[![Python](https://img.shields.io/pypi/pyversions/velixar)](https://pypi.org/project/velixar/)
[![License](https://img.shields.io/github/license/velixar/velixar-python)](LICENSE)

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

# Initialize client
v = Velixar(api_key="vlx_your_key")  # Or set VELIXAR_API_KEY env var

# Store a memory
memory_id = v.store(
    content="User prefers dark mode and metric units",
    tier=0,  # 0=pinned (critical), 2=semantic (default)
    user_id="user_123",
    tags=["preferences", "settings"],
)

# Search memories
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

# Store critical preference
v.store("User is allergic to peanuts", tier=MemoryTier.PINNED)

# Store session context
v.store("Currently discussing project X", tier=MemoryTier.SESSION)
```

## LangChain Integration

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from velixar.integrations.langchain import VelixarMemory

# Create memory backed by Velixar
memory = VelixarMemory(
    api_key="vlx_...",
    user_id="user_123",
)

# Use with any LangChain chain
chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
)

response = chain.invoke({"input": "Remember that I prefer Python over JavaScript"})
response = chain.invoke({"input": "What programming language do I prefer?"})
```

## LlamaIndex Integration

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from velixar.integrations.llamaindex import VelixarMemory

memory = VelixarMemory(api_key="vlx_...", user_id="user_123")

agent = ReActAgent.from_tools(
    tools=[...],
    llm=OpenAI(),
    memory=memory,
)
```

## OpenAI Function Calling

```python
from openai import OpenAI
from velixar import Velixar
from velixar.integrations.openai import VelixarAssistant

# Simple wrapper with automatic memory
assistant = VelixarAssistant(
    openai_client=OpenAI(),
    velixar_api_key="vlx_...",
    user_id="user_123",
)

assistant.chat("Remember that my birthday is March 15th")
assistant.chat("When is my birthday?")  # Uses memory automatically
```

## Batch Operations

```python
# Store multiple memories at once
result = v.store_many([
    {"content": "Fact 1", "tier": 0},
    {"content": "Fact 2", "tier": 2, "tags": ["important"]},
    {"content": "Fact 3", "user_id": "user_456"},
])
print(f"Stored {result.stored} memories")
```

## Error Handling

```python
from velixar import Velixar, VelixarError, RateLimitError, AuthenticationError

try:
    v = Velixar(api_key="invalid")
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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VELIXAR_API_KEY` | Your API key |
| `VELIXAR_BASE_URL` | Custom API endpoint |

## License

MIT License - see [LICENSE](LICENSE) for details.
