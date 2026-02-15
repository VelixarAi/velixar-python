"""OpenAI function calling integration."""

from __future__ import annotations

from typing import Any, Callable
from velixar import Velixar, MemoryTier


def get_velixar_tools(client: Velixar, user_id: str | None = None) -> list[dict[str, Any]]:
    """Get OpenAI function definitions for Velixar memory operations.
    
    Usage:
        from openai import OpenAI
        from velixar import Velixar
        from velixar.integrations.openai import get_velixar_tools, handle_velixar_call
        
        v = Velixar(api_key="vlx_...")
        tools = get_velixar_tools(v, user_id="user_123")
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
        )
        
        # Handle tool calls
        for call in response.choices[0].message.tool_calls:
            result = handle_velixar_call(v, call, user_id="user_123")
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Store important information to remember for later. Use for facts, preferences, or context the user wants you to remember.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember",
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["critical", "normal"],
                            "description": "How important this memory is. Critical = always remember, Normal = contextual",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search your memory for relevant information about a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def handle_velixar_call(
    client: Velixar,
    tool_call: Any,
    user_id: str | None = None,
) -> str:
    """Handle a Velixar tool call from OpenAI.
    
    Returns the result as a string for the assistant.
    """
    import json
    
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    if name == "remember":
        content = args["content"]
        importance = args.get("importance", "normal")
        tier = MemoryTier.PINNED if importance == "critical" else MemoryTier.SEMANTIC
        
        memory_id = client.store(content=content, user_id=user_id, tier=tier)
        return f"Remembered: {content[:50]}... (id: {memory_id})"
    
    elif name == "recall":
        query = args["query"]
        results = client.search(query=query, user_id=user_id, limit=5)
        
        if not results.memories:
            return "No relevant memories found."
        
        memories = "\n".join(f"- {m.content}" for m in results.memories)
        return f"Found {results.count} relevant memories:\n{memories}"
    
    return f"Unknown function: {name}"


class VelixarAssistant:
    """Wrapper for OpenAI assistants with Velixar memory.
    
    Usage:
        from openai import OpenAI
        from velixar.integrations.openai import VelixarAssistant
        
        assistant = VelixarAssistant(
            openai_client=OpenAI(),
            velixar_api_key="vlx_...",
            model="gpt-4",
        )
        
        response = assistant.chat("Remember that I prefer dark mode")
        response = assistant.chat("What are my preferences?")
    """

    def __init__(
        self,
        openai_client: Any,
        velixar_api_key: str | None = None,
        model: str = "gpt-4",
        user_id: str | None = None,
        system_prompt: str | None = None,
    ):
        self.openai = openai_client
        self.velixar = Velixar(api_key=velixar_api_key)
        self.model = model
        self.user_id = user_id
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with persistent memory. "
            "Use the 'remember' function to store important information the user tells you. "
            "Use the 'recall' function to search your memory when relevant."
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": message})
        
        tools = get_velixar_tools(self.velixar, self.user_id)
        
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls
        while assistant_message.tool_calls:
            self.messages.append(assistant_message.model_dump())
            
            for call in assistant_message.tool_calls:
                result = handle_velixar_call(self.velixar, call, self.user_id)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools,
            )
            assistant_message = response.choices[0].message
        
        content = assistant_message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        
        return content
