"""Basic usage examples for Velixar SDK."""

from velixar import Velixar, MemoryTier

# Initialize client
v = Velixar(api_key="vlx_your_key_here")

# ============================================================
# STORING MEMORIES
# ============================================================

# Simple store
memory_id = v.store("User prefers dark mode")
print(f"Stored memory: {memory_id}")

# Store with options
memory_id = v.store(
    content="User's favorite programming language is Python",
    user_id="user_123",
    tier=MemoryTier.PINNED,  # Critical fact, never expires
    tags=["preferences", "programming"],
    metadata={"source": "onboarding"},
)

# ============================================================
# SEARCHING MEMORIES
# ============================================================

# Basic search
results = v.search("programming preferences")
print(f"Found {results.count} memories")

for memory in results.memories:
    print(f"  [{memory.score:.2f}] {memory.content}")

# Search with filters
results = v.search(
    query="user settings",
    user_id="user_123",
    limit=5,
    tiers=[MemoryTier.PINNED, MemoryTier.SEMANTIC],
)

# ============================================================
# GET CONTEXT FOR LLM
# ============================================================

# Get formatted context string for prompts
context = v.get_context(
    query="What does the user prefer?",
    user_id="user_123",
    max_tokens=2000,
)

prompt = f"""Based on the following context about the user:

{context}

Answer the question: What are the user's preferences?"""

print(prompt)

# ============================================================
# BATCH OPERATIONS
# ============================================================

# Store multiple memories at once
result = v.store_many([
    {"content": "User lives in San Francisco", "tier": 0},
    {"content": "User works at a tech startup", "tier": 2},
    {"content": "User prefers morning meetings", "tags": ["scheduling"]},
])
print(f"Stored {result.stored} memories, {result.failed} failed")

# ============================================================
# MEMORY MANAGEMENT
# ============================================================

# Get specific memory
memory = v.get(memory_id)
print(f"Memory content: {memory.content}")

# Delete memory
deleted = v.delete(memory_id)
print(f"Deleted: {deleted}")

# ============================================================
# CONTEXT MANAGER
# ============================================================

# Auto-close client
with Velixar(api_key="vlx_...") as v:
    v.store("Temporary fact")
    results = v.search("fact")
# Client automatically closed
