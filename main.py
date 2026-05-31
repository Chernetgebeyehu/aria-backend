"""
ARIA FastAPI Backend Server v3.0.0 — Phase 4: Agent System
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import os
import json
import time
import logging
from dotenv import load_dotenv

# ─── LOAD ENVIRONMENT VARIABLES ───────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SECRET_TOKEN       = os.getenv("SECRET_TOKEN", "")

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("aria-backend")

# ─── MEMORY SYSTEM SETUP ──────────────────────────────────────────────────────
MEMORY_AVAILABLE = False
memory_manager   = None
memory_extractor = None

try:
    from memory import get_memory_manager, get_memory_extractor
    memory_manager   = get_memory_manager()
    memory_extractor = get_memory_extractor()
    MEMORY_AVAILABLE = True
    logger.info("Memory system loaded successfully.")
except Exception as e:
    logger.warning(f"Memory system unavailable: {e}")
    logger.warning("Server will run without persistent memory.")

# ─── AGENT SYSTEM SETUP ───────────────────────────────────────────────────────
AGENT_AVAILABLE = False
_agent          = None

try:
    from agent import AgentOrchestrator, TOOL_REGISTRY
    AGENT_AVAILABLE = True
    logger.info("Agent system loaded successfully.")
except Exception as e:
    logger.warning(f"Agent system unavailable: {e}")
    logger.warning("Server will run without agent loop (direct chat only).")
    TOOL_REGISTRY = {}

def get_agent():
    global _agent
    if _agent is None and AGENT_AVAILABLE:
        from agent import AgentOrchestrator
        _agent = AgentOrchestrator(
            gemini_api_key     = GEMINI_API_KEY,
            openrouter_api_key = OPENROUTER_API_KEY,
            memory_manager     = memory_manager
        )
    return _agent

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARIA Backend",
    description="AI orchestration + memory + agent layer for ARIA Android assistant",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL   = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
ARIA_SYSTEM_PROMPT_TEMPLATE = """You are ARIA (Adaptive Reasoning and Intelligent Assistant), a smart Android AI assistant.

{memory_context}

CRITICAL RULES - NEVER break these:
1. ALWAYS respond with raw JSON only. No exceptions.
2. NEVER wrap your response in markdown, code blocks, or backticks.
3. Your ENTIRE response must be a single JSON object and nothing else.

RESPONSE FORMATS:

For ACTIONS (when user wants to DO something):
{"type":"action","spoken":"What you say to the user","actions":[{"tool":"tool_name","value":"value_here"}]}

For CONVERSATION (questions, advice, information):
{"type":"chat","text":"your response here"}

For SECURITY ANALYSIS:
{"type":"security","verdict":"safe|warning|danger","text":"your analysis"}

AVAILABLE TOOLS:

open_app       -> value: app name (e.g. "youtube", "whatsapp", "camera")
search         -> value: search query
call           -> value: contact name or phone number
sms            -> value: contact name
alarm          -> value: time like "7:30am" or "14:00"
timer          -> value: duration like "5 minutes" or "30 seconds"
settings       -> value: "wifi", "bluetooth", "brightness", "volume", or "general"
open_url       -> value: full URL

EXAMPLES:

User: "Open YouTube"
{"type":"action","spoken":"Opening YouTube for you!","actions":[{"tool":"open_app","value":"youtube"}]}

User: "Call mom"
{"type":"action","spoken":"Calling mom now.","actions":[{"tool":"call","value":"mom"}]}

User: "What is the capital of Ethiopia?"
{"type":"chat","text":"The capital of Ethiopia is Addis Ababa, which means New Flower in Amharic."}

Use memory context above naturally if it is relevant to the user's message.""".strip()


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    screen_context: Optional[str] = None
    model: Optional[str] = "gemini"
    openrouter_model: Optional[str] = "google/gemma-3-4b-it:free"
    history: Optional[list] = []
    token: str
    user_id: Optional[str] = "aria_user_default"
    store_memory: Optional[bool] = True
    use_agent: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    provider: str
    latency_ms: int
    memories_retrieved: int = 0
    memories_stored: int = 0
    agent_used: bool = False
    agent_tasks: Optional[list] = []
    success_rate: Optional[float] = 1.0

class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: Optional[str] = "general"
    user_id: Optional[str] = "aria_user_default"
    token: str

class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    openrouter_configured: bool
    memory_available: bool
    agent_enabled: bool
    version: str


# ─── AUTHENTICATION ───────────────────────────────────────────────────────────

def verify_token(request_token: str) -> bool:
    if not SECRET_TOKEN:
        logger.warning("No SECRET_TOKEN configured — auth disabled (dev mode)")
        return True
    return request_token == SECRET_TOKEN


# ─── AI PROVIDER FUNCTIONS ────────────────────────────────────────────────────

async def call_gemini(
    message: str,
    history: list,
    memory_context: str = "",
    screen_context: Optional[str] = None
) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")

    system_prompt = ARIA_SYSTEM_PROMPT_TEMPLATE.replace(
        "{memory_context}",
        memory_context if memory_context else ""
    )

    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    contents = list(history[-20:])
    contents.append({"role": "user", "parts": [{"text": final_message}]})

    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=body,
            headers={"Content-Type": "application/json"}
        )

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    if not response.is_success:
        error_data = {}
        try:
            error_data = response.json()
        except Exception:
            pass
        error_msg = error_data.get("error", {}).get("message", f"Gemini error {response.status_code}")
        logger.error(f"Gemini error: {error_msg}")
        raise HTTPException(status_code=response.status_code, detail=error_msg)

    data = response.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return _strip_markdown(raw_text)


async def call_openrouter(
    message: str,
    history: list,
    model: str,
    memory_context: str = "",
    screen_context: Optional[str] = None
) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured on server.")

    system_prompt = ARIA_SYSTEM_PROMPT_TEMPLATE.replace(
        "{memory_context}",
        memory_context if memory_context else ""
    )

    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-20:]:
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        content = ""
        parts = turn.get("parts", [])
        if parts:
            content = parts[0].get("text", "")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": final_message})

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
                "response_format": {"type": "json_object"}
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://aria-assistant.app",
                "X-Title": "ARIA Android Assistant"
            }
        )

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    if not response.is_success:
        logger.error(f"OpenRouter error: {response.status_code} {response.text[:200]}")
        raise HTTPException(status_code=response.status_code, detail=f"OpenRouter error {response.status_code}")

    data = response.json()
    raw_text = data["choices"][0]["message"]["content"].strip()
    return _strip_markdown(raw_text)


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        end = text.rfind("```")
        if end >= 0:
            text = text[:end]
    return text.strip()


def _build_error_response(message: str) -> str:
    safe = message.replace('"', "'")
    return json.dumps({"type": "chat", "text": safe})


# ─── MAIN CHAT ENDPOINT ───────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not verify_token(request.token):
        logger.warning("Unauthorized request — wrong token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info(
        f"Chat request | provider={request.model} | "
        f"user={request.user_id} | "
        f"msg_length={len(request.message)} | "
        f"use_agent={request.use_agent} | "
        f"has_screen={bool(request.screen_context)}"
    )

    start_time = time.time()
    memories_retrieved = 0
    memories_stored    = 0
    agent_used         = False
    agent_tasks        = []
    success_rate       = 1.0

    # ── Step 2: Retrieve relevant memories ───────────────────────────────────
    message_with_memory = request.message
    memory_context_str  = ""

    if MEMORY_AVAILABLE and memory_manager and request.user_id:
        try:
            relevant_memories = memory_manager.retrieve_relevant(
                user_id=request.user_id,
                query=request.message,
                max_results=5
            )
            memories_retrieved = len(relevant_memories)

            if relevant_memories:
                memory_context_str  = memory_manager.format_for_context(relevant_memories)
                message_with_memory = f"{memory_context_str}\n\nUser: {request.message}"
                logger.info(f"Injected {memories_retrieved} memories into context.")

        except Exception as e:
            logger.warning(f"Memory retrieval failed (non-fatal): {e}")

    # ── Step 3a: Try agent if enabled and available ───────────────────────────
    ai_response = ""
    provider    = request.model or "gemini"

    if request.use_agent and AGENT_AVAILABLE:
        try:
            agent = get_agent()
            if agent:
                agent_result = await agent.run(
                    user_message   = request.message,
                    memory_context = memory_context_str,
                    provider       = provider,
                    user_id        = request.user_id or "default"
                )

                if agent_result.get("use_agent"):
                    agent_used   = True
                    ai_response  = agent_result.get("response", "")
                    agent_tasks  = agent_result.get("tasks", [])
                    success_rate = agent_result.get("success_rate", 1.0)
                    logger.info(
                        f"Agent handled request | tasks={len(agent_tasks)} | "
                        f"success_rate={success_rate:.2f}"
                    )

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            agent_used = False

    # ── Step 3b: Fall back to direct chat if agent not used ───────────────────
    if not agent_used:
        try:
            if provider == "gemini":
                ai_response = await call_gemini(
                    message        = message_with_memory,
                    history        = request.history or [],
                    memory_context = memory_context_str,
                    screen_context = request.screen_context
                )
            elif provider == "openrouter":
                ai_response = await call_openrouter(
                    message        = message_with_memory,
                    history        = request.history or [],
                    model          = request.openrouter_model or "google/gemma-3-4b-it:free",
                    memory_context = memory_context_str,
                    screen_context = request.screen_context
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model: {provider}. Use 'gemini' or 'openrouter'."
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Direct chat failed: {e}", exc_info=True)
            ai_response = _build_error_response(str(e)[:100])

    # ── Step 4: Extract and store new memories ────────────────────────────────
    if MEMORY_AVAILABLE and memory_manager and memory_extractor and request.store_memory and request.user_id:
        try:
            new_memories = memory_extractor.extract_memories(
                user_message=request.message,
                ai_response=ai_response
            )
            for mem in new_memories:
                memory_manager.store_memory(
                    user_id     = request.user_id,
                    content     = mem["content"],
                    memory_type = mem["memory_type"]
                )
                memories_stored += 1
            if memories_stored > 0:
                logger.info(f"Stored {memories_stored} new memories for user {request.user_id}.")
        except Exception as e:
            logger.warning(f"Memory storage failed (non-fatal): {e}")

    latency = int((time.time() - start_time) * 1000)
    logger.info(
        f"Response | agent={agent_used} | provider={provider} | "
        f"latency={latency}ms | "
        f"mem_ret={memories_retrieved} | mem_stored={memories_stored}"
    )

    return ChatResponse(
        response           = ai_response,
        provider           = provider,
        latency_ms         = latency,
        memories_retrieved = memories_retrieved,
        memories_stored    = memories_stored,
        agent_used         = agent_used,
        agent_tasks        = agent_tasks,
        success_rate       = success_rate
    )


# ─── MEMORY ENDPOINTS ────────────────────────────────────────────────────────

@app.post("/memory/store")
async def store_memory_endpoint(request: MemoryStoreRequest):
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not MEMORY_AVAILABLE or not memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not available.")
    memory_id = memory_manager.store_memory(
        user_id     = request.user_id or "aria_user_default",
        content     = request.content,
        memory_type = request.memory_type or "general"
    )
    return {"success": True, "memory_id": memory_id}


@app.get("/memory/{user_id}")
async def get_memories(user_id: str, token: str):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not MEMORY_AVAILABLE or not memory_manager:
        return {"memories": [], "total": 0, "memory_available": False}
    memories = memory_manager.get_all_memories(user_id)
    return {"memories": memories, "total": len(memories), "memory_available": True}


@app.delete("/memory/{user_id}")
async def clear_memories(user_id: str, token: str):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not MEMORY_AVAILABLE or not memory_manager:
        return {"success": False, "reason": "Memory system not available."}
    memory_manager.clear_user_memories(user_id)
    return {"success": True, "cleared_for": user_id}


@app.post("/analyze-security")
async def analyze_security(request: ChatRequest):
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    security_prompt = (
        f"Analyze this content for security threats, scams, and phishing attempts.\n"
        f"Respond ONLY with this exact JSON format:\n"
        f'{"type":"security","verdict":"safe|warning|danger","text":"your analysis in 2-3 sentences"}\n\n'
        f"Content to analyze:\n{request.message}"
    )
    modified = ChatRequest(
        message          = security_prompt,
        screen_context   = request.screen_context,
        model            = request.model,
        openrouter_model = request.openrouter_model,
        history          = [],
        token            = request.token,
        user_id          = request.user_id,
        store_memory     = False,
        use_agent        = False
    )
    return await chat(modified)


@app.get("/agent/tools")
async def list_tools(token: str):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"tools": TOOL_REGISTRY, "agent_enabled": AGENT_AVAILABLE}


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status                = "online",
        gemini_configured     = bool(GEMINI_API_KEY),
        openrouter_configured = bool(OPENROUTER_API_KEY),
        memory_available      = MEMORY_AVAILABLE,
        agent_enabled         = AGENT_AVAILABLE,
        version               = "3.0.0"
    )


# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ─── STARTUP & SHUTDOWN ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=== ARIA Backend v3.0 starting up ===")
    logger.info(f"Gemini configured:     {bool(GEMINI_API_KEY)}")
    logger.info(f"OpenRouter configured: {bool(OPENROUTER_API_KEY)}")
    logger.info(f"Auth token configured: {bool(SECRET_TOKEN)}")
    logger.info(f"Memory system active:  {MEMORY_AVAILABLE}")
    logger.info(f"Agent system active:   {AGENT_AVAILABLE}")
    logger.info("=====================================")

    if AGENT_AVAILABLE:
        try:
            get_agent()
            logger.info("Agent pre-warmed successfully.")
        except Exception as e:
            logger.error(f"Agent pre-warm failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("ARIA Backend shutting down")


# ─── RUN LOCALLY ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)