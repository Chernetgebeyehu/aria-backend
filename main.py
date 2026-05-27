"""
ARIA FastAPI Backend Server — Phase 2 + Phase 3 (Memory System)
================================================================

Think of this file as a telephone switchboard operator:
- Android phones call in with messages
- This server retrieves relevant memories from the past
- Injects those memories into the AI's context
- Calls the right AI (Gemini or OpenRouter)
- Extracts new memories from the conversation
- Sends the response back to the phone

The server NEVER exposes AI API keys to the phone.
The phone only needs to know this server's URL + secret token.

What changed from Phase 2:
- memory.py is now ACTUALLY connected and used
- /chat endpoint now reads and stores memories automatically
- /memory/store lets the app save a memory explicitly
- /memory/{user_id} lets the app show what ARIA remembers
- ChatRequest now has user_id and store_memory fields
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
# This reads your .env file locally, and Railway env vars in production.
# Like opening your safe and putting the contents on your desk (privately).
load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SECRET_TOKEN       = os.getenv("SECRET_TOKEN", "")

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
# Logging = writing a diary of everything that happens.
# When something breaks at 3am, you read the diary to find out why.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("aria-backend")

# ─── MEMORY SYSTEM SETUP ──────────────────────────────────────────────────────
# We try to load the memory system.
# If it fails (ChromaDB not installed, etc.), we run WITHOUT memory.
# This means the server still works — it just won't remember anything.
# Think of it like a person who works fine even with amnesia —
# they can still answer questions, they just won't remember past conversations.

MEMORY_AVAILABLE = False
memory_manager = None
memory_extractor = None

try:
    from memory import get_memory_manager, get_memory_extractor
    memory_manager = get_memory_manager()
    memory_extractor = get_memory_extractor()
    MEMORY_AVAILABLE = True
    logger.info("✅ Memory system loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️  Memory system unavailable: {e}")
    logger.warning("    Server will run without persistent memory.")

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────
# This creates the web server application.
# Think of it as building the building before putting rooms in it.
app = FastAPI(
    title="ARIA Backend",
    description="Secure AI orchestration layer for ARIA Android assistant",
    version="2.0.0"
)

# ─── CORS MIDDLEWARE ──────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
# Without this, browsers block requests from unknown sources.
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
# This is what tells the AI how to behave as ARIA.
# Living on the server means you can update ARIA's personality
# without releasing a new version of the Android app.
# That's a huge engineering advantage.
ARIA_SYSTEM_PROMPT = """
You are ARIA (Adaptive Reasoning and Intelligent Assistant), a smart Android AI assistant.

CRITICAL RULES - NEVER break these:
1. ALWAYS respond with raw JSON only. No exceptions.
2. NEVER wrap your response in markdown, code blocks, or backticks.
3. Your ENTIRE response must be a single JSON object and nothing else.

═══ RESPONSE FORMATS ═══

For ACTIONS (when user wants to DO something):
{"type":"action","spoken":"What you say to the user","actions":[{"tool":"tool_name","value":"value_here"}]}

For CONVERSATION (questions, advice, information):
{"type":"chat","text":"your response here"}

For SECURITY ANALYSIS:
{"type":"security","verdict":"safe|warning|danger","text":"your analysis"}

═══ AVAILABLE TOOLS ═══

open_app       → value: app name (e.g. "youtube", "whatsapp", "camera")
search         → value: search query
call           → value: contact name or phone number
sms            → value: contact name
alarm          → value: time like "7:30am" or "14:00"
timer          → value: duration like "5 minutes" or "30 seconds"
settings       → value: "wifi", "bluetooth", "brightness", "volume", or "general"
open_url       → value: full URL

═══ EXAMPLES ═══

User: "Open YouTube"
{"type":"action","spoken":"Opening YouTube for you!","actions":[{"tool":"open_app","value":"youtube"}]}

User: "Call mom"
{"type":"action","spoken":"Calling mom now.","actions":[{"tool":"call","value":"mom"}]}

User: "What is the capital of Ethiopia?"
{"type":"chat","text":"The capital of Ethiopia is Addis Ababa (አዲስ አበባ), which means New Flower in Amharic."}
""".strip()


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────
# Pydantic models define exactly what shape data must be in.
# If the phone sends wrong data, FastAPI automatically rejects it
# with a clear error message. Like a bouncer checking IDs at the door.

class ChatRequest(BaseModel):
    """What the Android app sends to the server."""
    message: str                              # The user's text/voice input
    screen_context: Optional[str] = None      # Current screen content (for safety checks)
    model: Optional[str] = "gemini"           # Which AI to use: "gemini" or "openrouter"
    openrouter_model: Optional[str] = "google/gemma-3-4b-it:free"
    history: Optional[list] = []              # Recent conversation turns (short-term)
    token: str                                # Secret token — must match SECRET_TOKEN
    user_id: Optional[str] = "aria_user_default"   # Which user this is (for memory)
    store_memory: Optional[bool] = True       # Whether to save memories from this chat

class ChatResponse(BaseModel):
    """What the server sends back to the Android app."""
    response: str           # The AI's JSON response
    provider: str           # Which AI was used ("gemini" or "openrouter")
    latency_ms: int         # How long it took in milliseconds
    memories_retrieved: int = 0   # How many memories were used as context
    memories_stored: int = 0      # How many new memories were saved

class MemoryStoreRequest(BaseModel):
    """Request to explicitly store a memory."""
    content: str
    memory_type: Optional[str] = "general"
    user_id: Optional[str] = "aria_user_default"
    token: str

class HealthResponse(BaseModel):
    """Status check response."""
    status: str
    gemini_configured: bool
    openrouter_configured: bool
    memory_available: bool
    version: str


# ─── AUTHENTICATION ───────────────────────────────────────────────────────────
# This checks every request to make sure it comes from our Android app.
# Your server URL is public — anyone can find it.
# Without this token, anyone could use your server and drain your API credits.

def verify_token(request_token: str) -> bool:
    """
    Verify the secret token from the Android app.
    If no SECRET_TOKEN is configured in env vars, auth is disabled (dev mode).
    """
    if not SECRET_TOKEN:
        logger.warning("No SECRET_TOKEN configured — auth disabled (dev mode)")
        return True
    return request_token == SECRET_TOKEN


# ─── AI PROVIDER FUNCTIONS ────────────────────────────────────────────────────

async def call_gemini(
    message: str,
    history: list,
    screen_context: Optional[str] = None
) -> str:
    """
    Call Google's Gemini AI.

    'async' means this function can pause while waiting for the network
    and let other requests be handled in the meantime.
    Like a waiter who takes 5 orders before going to the kitchen,
    instead of waiting for one meal to be cooked before taking the next.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")

    # Build the final message — attach screen context if we have it
    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    # Build conversation history in Gemini format
    # We only send the last 20 items (10 turns) to save tokens
    contents = []
    for turn in history[-20:]:
        contents.append(turn)

    # Add the new user message
    contents.append({
        "role": "user",
        "parts": [{"text": final_message}]
    })

    # Build the full request body
    body = {
        "systemInstruction": {
            "parts": [{"text": ARIA_SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json"
        }
    }

    # Make the HTTP request to Gemini
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

    # Parse the response — dig into the nested JSON structure Gemini returns
    data = response.json()
    raw_text = (
        data["candidates"][0]["content"]["parts"][0]["text"]
        .strip()
    )
    return strip_markdown(raw_text)


async def call_openrouter(
    message: str,
    history: list,
    model: str,
    screen_context: Optional[str] = None
) -> str:
    """
    Call OpenRouter — a service that gives access to many AI models
    through one single API. Like a broker who can connect you to
    hundreds of AI models from one place (Gemma, Llama, Mistral, Claude, etc.)
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured on server.")

    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    # OpenRouter uses OpenAI's message format (role: user/assistant/system)
    messages = [{"role": "system", "content": ARIA_SYSTEM_PROMPT}]

    for turn in history[-20:]:
        # Convert from Gemini format (model) to OpenRouter format (assistant)
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        content = ""
        parts = turn.get("parts", [])
        if parts:
            content = parts[0].get("text", "")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": final_message})

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            json=body,
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
    return strip_markdown(raw_text)


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """
    Remove markdown code fences if the AI wrapped the JSON in them.
    Some models do this even when told not to — this is the safety net.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        end = text.rfind("```")
        if end >= 0:
            text = text[:end]
    return text.strip()


def build_error_response(message: str) -> str:
    """Build a safe JSON error response in ARIA's chat format."""
    safe = message.replace('"', "'")
    return json.dumps({"type": "chat", "text": safe})


# ─── API ENDPOINTS ────────────────────────────────────────────────────────────
# These are the "rooms" in your server building.
# Each endpoint is a URL your Android app can call.

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Like a "Are you there?" ping.
    URL: GET https://your-server.com/
    """
    return HealthResponse(
        status="online",
        gemini_configured=bool(GEMINI_API_KEY),
        openrouter_configured=bool(OPENROUTER_API_KEY),
        memory_available=MEMORY_AVAILABLE,
        version="2.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. This is what your Android app calls for every message.

    The full flow with memory:
    1. Verify the secret token
    2. Retrieve relevant past memories for this user
    3. Inject those memories into the message context
    4. Call the right AI provider (Gemini or OpenRouter)
    5. Extract new memories from what the user just said
    6. Return the AI response + memory stats to the phone

    URL: POST https://your-server.com/chat
    """

    # ── STEP 1: Verify token ──────────────────────────────────────────────────
    if not verify_token(request.token):
        logger.warning("Unauthorized request — wrong token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info(
        f"Chat request | provider={request.model} | "
        f"user={request.user_id} | "
        f"msg_length={len(request.message)} | "
        f"has_screen={bool(request.screen_context)}"
    )

    start_time = time.time()
    memories_retrieved = 0
    memories_stored = 0

    # ── STEP 2: Retrieve relevant memories ───────────────────────────────────
    # This is RAG (Retrieval-Augmented Generation).
    # We search ARIA's long-term memory for anything related to this message.
    # Example: User says "remind me of my meeting" →
    #   Memory system finds "User has a meeting on Monday at 3pm" →
    #   That memory gets added to the message context →
    #   AI responds knowing about the meeting.
    message_with_memory = request.message

    if MEMORY_AVAILABLE and memory_manager and request.user_id:
        try:
            relevant_memories = memory_manager.retrieve_relevant(
                user_id=request.user_id,
                query=request.message,
                max_results=5
            )
            memories_retrieved = len(relevant_memories)

            if relevant_memories:
                # Format memories as text and prepend to the message
                # The AI sees this as part of the user's message context
                memory_context = memory_manager.format_for_context(relevant_memories)
                if memory_context:
                    message_with_memory = f"{memory_context}\n\nUser: {request.message}"
                    logger.info(f"Injected {memories_retrieved} memories into context.")

        except Exception as e:
            logger.warning(f"Memory retrieval failed (non-fatal): {e}")

    # ── STEP 3: Call the AI ───────────────────────────────────────────────────
    try:
        if request.model == "gemini":
            ai_response = await call_gemini(
                message=message_with_memory,
                history=request.history or [],
                screen_context=request.screen_context
            )
            provider = "gemini"

        elif request.model == "openrouter":
            ai_response = await call_openrouter(
                message=message_with_memory,
                history=request.history or [],
                model=request.openrouter_model or "google/gemma-3-4b-it:free",
                screen_context=request.screen_context
            )
            provider = "openrouter"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}. Use 'gemini' or 'openrouter'.")

    except HTTPException:
        raise   # Re-raise HTTP exceptions as-is

    except Exception as e:
        logger.error(f"Unexpected AI error: {e}", exc_info=True)
        ai_response = build_error_response(f"Server error: {str(e)[:100]}")
        provider = "error"

    # ── STEP 4: Extract and store new memories ────────────────────────────────
    # After the conversation, we look at what the user said and ask:
    # "Is there anything worth remembering here?"
    # Example: "My name is Cherinet" → worth storing as a fact.
    # Example: "Open YouTube" → not worth storing.
    if MEMORY_AVAILABLE and memory_manager and memory_extractor and request.store_memory and request.user_id:
        try:
            new_memories = memory_extractor.extract_memories(
                user_message=request.message,
                ai_response=ai_response
            )
            for mem in new_memories:
                memory_manager.store_memory(
                    user_id=request.user_id,
                    content=mem["content"],
                    memory_type=mem["memory_type"]
                )
                memories_stored += 1

            if memories_stored > 0:
                logger.info(f"Stored {memories_stored} new memories for user {request.user_id}.")

        except Exception as e:
            logger.warning(f"Memory storage failed (non-fatal): {e}")

    latency = int((time.time() - start_time) * 1000)
    logger.info(f"Response sent | provider={provider} | latency={latency}ms | "
                f"memories_retrieved={memories_retrieved} | memories_stored={memories_stored}")

    return ChatResponse(
        response=ai_response,
        provider=provider,
        latency_ms=latency,
        memories_retrieved=memories_retrieved,
        memories_stored=memories_stored
    )


@app.post("/memory/store")
async def store_memory_endpoint(request: MemoryStoreRequest):
    """
    Explicitly store a memory from the app.
    This is called when the user taps "Remember this" on a message.
    URL: POST https://your-server.com/memory/store
    """
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not MEMORY_AVAILABLE or not memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not available.")

    memory_id = memory_manager.store_memory(
        user_id=request.user_id or "aria_user_default",
        content=request.content,
        memory_type=request.memory_type or "general"
    )

    return {"success": True, "memory_id": memory_id}


@app.get("/memory/{user_id}")
async def get_memories(user_id: str, token: str):
    """
    Get all stored memories for a user.
    The app calls this to show the memory list in settings.
    URL: GET https://your-server.com/memory/{user_id}?token=your_token
    """
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not MEMORY_AVAILABLE or not memory_manager:
        return {"memories": [], "total": 0, "memory_available": False}

    memories = memory_manager.get_all_memories(user_id)
    return {
        "memories": memories,
        "total": len(memories),
        "memory_available": True
    }


@app.delete("/memory/{user_id}")
async def clear_memories(user_id: str, token: str):
    """
    Delete all memories for a user.
    The app calls this when user taps "Clear ARIA's memory."
    URL: DELETE https://your-server.com/memory/{user_id}?token=your_token
    """
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not MEMORY_AVAILABLE or not memory_manager:
        return {"success": False, "reason": "Memory system not available."}

    memory_manager.clear_user_memories(user_id)
    return {"success": True, "cleared_for": user_id}


@app.post("/analyze-security")
async def analyze_security(request: ChatRequest):
    """
    Dedicated security analysis endpoint.
    Sends screen content to AI specifically for security checking.
    URL: POST https://your-server.com/analyze-security
    """
    if not verify_token(request.token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    security_prompt = f"""
Analyze this content for security threats, scams, and phishing attempts.
Respond ONLY with this exact JSON format:
{{"type":"security","verdict":"safe|warning|danger","text":"your analysis in 2-3 sentences"}}

Content to analyze:
{request.message}
""".strip()

    modified_request = ChatRequest(
        message=security_prompt,
        screen_context=request.screen_context,
        model=request.model,
        openrouter_model=request.openrouter_model,
        history=[],
        token=request.token,
        user_id=request.user_id,
        store_memory=False   # Don't store memories from security scans
    )

    return await chat(modified_request)


# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────
# What to do when things go wrong.
# Like having a fire escape plan — you hope you never need it,
# but you are VERY glad it's there when you do.

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ─── STARTUP & SHUTDOWN ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=== ARIA Backend v2.0 starting up ===")
    logger.info(f"Gemini configured:    {bool(GEMINI_API_KEY)}")
    logger.info(f"OpenRouter configured: {bool(OPENROUTER_API_KEY)}")
    logger.info(f"Auth token configured: {bool(SECRET_TOKEN)}")
    logger.info(f"Memory system active:  {MEMORY_AVAILABLE}")
    logger.info("=====================================")

@app.on_event("shutdown")
async def shutdown():
    logger.info("ARIA Backend shutting down")


# ─── RUN LOCALLY ──────────────────────────────────────────────────────────────
# This runs only when you execute `python main.py` directly on your PC.
# On Railway, the Procfile handles starting the server instead.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True   # Auto-restart when you save changes (dev mode only!)
    )