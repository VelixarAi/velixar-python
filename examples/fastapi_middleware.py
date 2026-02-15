"""FastAPI middleware for automatic memory context."""

from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

from velixar import Velixar, AsyncVelixar

app = FastAPI()

# Initialize Velixar client
velixar = AsyncVelixar(api_key="vlx_your_key")


# ============================================================
# MIDDLEWARE - Auto-inject memory context
# ============================================================

class VelixarMiddleware:
    """Middleware that attaches memory context to requests."""
    
    def __init__(self, app: FastAPI, velixar_client: AsyncVelixar):
        self.app = app
        self.velixar = velixar_client
    
    async def __call__(self, request: Request, call_next):
        # Extract user ID from auth header or session
        user_id = request.headers.get("X-User-ID")
        
        if user_id:
            # Attach velixar client to request state
            request.state.velixar = self.velixar
            request.state.user_id = user_id
        
        response = await call_next(request)
        return response


# Add middleware
app.add_middleware(VelixarMiddleware, velixar_client=velixar)


# ============================================================
# DEPENDENCY - Get memory context
# ============================================================

async def get_memory_context(request: Request, query: str = "") -> str:
    """Dependency that retrieves relevant memory context."""
    if not hasattr(request.state, "velixar"):
        return ""
    
    if not query:
        return ""
    
    results = await request.state.velixar.search(
        query=query,
        user_id=request.state.user_id,
        limit=5,
    )
    
    return "\n".join(m.content for m in results.memories)


# ============================================================
# ROUTES
# ============================================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    memories_used: int


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
):
    """Chat endpoint with automatic memory retrieval."""
    
    user_id = getattr(request.state, "user_id", None)
    v = getattr(request.state, "velixar", None)
    
    memories_used = 0
    context = ""
    
    if v and user_id:
        # Get relevant memories
        results = await v.search(body.message, user_id=user_id, limit=5)
        memories_used = results.count
        context = "\n".join(m.content for m in results.memories)
    
    # Build prompt with context
    prompt = f"""Context from memory:
{context}

User message: {body.message}

Respond helpfully:"""
    
    # Call your LLM here
    response = f"[Would call LLM with context: {memories_used} memories]"
    
    # Store the interaction
    if v and user_id:
        await v.store(
            content=f"User asked: {body.message}",
            user_id=user_id,
            tier=1,  # Session tier
        )
    
    return ChatResponse(response=response, memories_used=memories_used)


@app.post("/remember")
async def remember(
    request: Request,
    content: str,
    tier: int = 2,
):
    """Explicitly store a memory."""
    user_id = getattr(request.state, "user_id", None)
    v = getattr(request.state, "velixar", None)
    
    if not v or not user_id:
        return {"error": "Not authenticated"}
    
    memory_id = await v.store(
        content=content,
        user_id=user_id,
        tier=tier,
    )
    
    return {"id": memory_id, "stored": True}


@app.get("/memories")
async def get_memories(
    request: Request,
    query: str = "",
    limit: int = 10,
):
    """Search user's memories."""
    user_id = getattr(request.state, "user_id", None)
    v = getattr(request.state, "velixar", None)
    
    if not v or not user_id:
        return {"error": "Not authenticated"}
    
    results = await v.search(
        query=query or "*",
        user_id=user_id,
        limit=limit,
    )
    
    return {
        "count": results.count,
        "memories": [
            {"id": m.id, "content": m.content, "score": m.score}
            for m in results.memories
        ],
    }


# ============================================================
# STARTUP/SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown():
    await velixar.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
