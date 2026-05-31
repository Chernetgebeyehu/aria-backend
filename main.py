"""
ARIA FastAPI Backend Server v3.1.0
====================================
What changed from v3.0.1:
  - Real-time knowledge via Gemini Google Search grounding
  - Smart routing: only grounded calls for queries that need live data
  - Two-call architecture preserves the existing JSON contract with Android
  - Grounding metadata (sources, citations) passed back to Android
  - Weather no longer just opens a browser — ARIA actually answers it
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
import re
import logging
from datetime import datetime, timezone
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

# ─── AGENT SYSTEM SETUP ───────────────────────────────────────────────────────
AGENT_AVAILABLE = False
_agent          = None

try:
    from agent import AgentOrchestrator, TOOL_REGISTRY
    AGENT_AVAILABLE = True
    logger.info("Agent system loaded successfully.")
except Exception as e:
    logger.warning(f"Agent system unavailable: {e}")
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
    description="AI orchestration + memory + agent + real-time search for ARIA Android assistant",
    version="3.1.0"
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

# ─── REAL-TIME SEARCH ROUTING ─────────────────────────────────────────────────
#
# This is the classifier that decides whether a message needs Gemini's
# Google Search grounding tool or can be answered from training data alone.
#
# Why this matters:
#   Gemini grounding adds ~500-800ms latency and cannot be combined with
#   responseMimeType="application/json". So we only activate it when the
#   query genuinely needs live data. Everything else uses the fast JSON path.
#
# How it works:
#   We check two things:
#   1. Does the message contain a real-time signal keyword?
#      (weather, news, price, score, today, right now, latest, etc.)
#   2. Is the message NOT a device action?
#      ("open youtube" needs the agent, not a web search)
#
# This keeps ARIA's response format intact for 95% of requests while
# giving real answers for the 5% that genuinely need live data.

REALTIME_SIGNALS = [
    # Weather
    "weather", "forecast", "temperature", "rain", "sunny", "humidity",
    "wind speed", "celsius", "fahrenheit",
    # News & current events
    "news", "latest", "breaking", "just happened", "recently",
    "current events", "what happened", "today's",
    # Time-sensitive facts
    "right now", "at the moment", "currently", "as of today",
    "live score", "match score", "game score", "standings",
    # Prices & markets
    "price of", "stock price", "exchange rate", "bitcoin", "crypto",
    "dollar", "euro", "birr", "how much is",
    # Sports
    "who won", "score today", "match today", "fixture", "result",
    # People & organizations (current state)
    "who is the current", "who is the president", "who is the prime minister",
    "who is the ceo", "who leads", "current leader",
    # Ethiopia-specific real-time info
    "addis ababa weather", "ethiopia news", "ethiopian",
]

# These prefixes mean the user wants ARIA to DO something on the device.
# They go through the agent, NOT web search.
ACTION_PREFIXES = [
    "open ", "call ", "send ", "set alarm", "set timer",
    "play ", "launch ", "turn on", "turn off", "message ",
    "remind me", "search for",   # "search for" → device search, not grounding
]

def needs_realtime_search(message: str) -> bool:
    """
    Decide if a message needs Gemini Google Search grounding.

    Returns True only when:
      - The message contains a real-time signal keyword
      - AND it is NOT a device action command

    This is intentionally conservative. False negatives (grounding missed)
    are better than false positives (JSON mode broken for action commands).
    """
    lower = message.lower().strip()

    # Never use grounding for device actions — they go through the agent
    for prefix in ACTION_PREFIXES:
        if lower.startswith(prefix):
            return False

    # Check for real-time signal keywords
    for signal in REALTIME_SIGNALS:
        if signal in lower:
            logger.info(f"Real-time signal detected: '{signal}' → using grounding")
            return True

    return False


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
ARIA_SYSTEM_PROMPT_TEMPLATE = """You are ARIA (Adaptive Reasoning and Intelligent Assistant), a smart Android AI assistant.

TODAY'S DATE: {current_date}
Always use this date when the user asks what year, month, or date it is. Never say you don't know the current date.

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

# ─── GROUNDED SYSTEM PROMPT ───────────────────────────────────────────────────
#
# This is a SEPARATE system prompt used ONLY when grounding is active.
#
# Why different from the main prompt?
# When grounding is enabled, Gemini returns plain text — not JSON. We cannot
# use responseMimeType="application/json" with grounding. So this prompt asks
# Gemini to respond naturally as a helpful assistant, and we wrap the result
# in {"type":"chat","text":"..."} ourselves after the call returns.
#
# The main system prompt's JSON rules are removed here because they conflict
# with how grounding works — grounding adds citation metadata to the response
# that cannot fit inside a strict JSON structure.

ARIA_GROUNDED_PROMPT_TEMPLATE = """You are ARIA, a smart Android AI assistant helping a user in real time.

TODAY'S DATE: {current_date}

{memory_context}

You have access to Google Search to answer questions about current events,
weather, news, prices, sports scores, and anything that requires up-to-date information.

Instructions:
- Answer the user's question directly and concisely.
- Use the search results to give accurate, current information.
- If the search returned weather data, include temperature, conditions, and a brief forecast.
- If the search returned news, summarize the key facts clearly.
- Keep answers under 3 sentences unless more detail is genuinely needed.
- Do NOT mention that you searched or that you have grounding — just answer naturally.
- Do NOT use markdown formatting — plain text only, since this will be spoken aloud.""".strip()


def build_system_prompt(memory_context: str = "") -> str:
    """Standard JSON-mode system prompt — used for all non-grounded calls."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%A, %B %d, %Y")
    return ARIA_SYSTEM_PROMPT_TEMPLATE.replace(
        "{current_date}", date_str
    ).replace(
        "{memory_context}", memory_context if memory_context else ""
    )


def build_grounded_prompt(memory_context: str = "") -> str:
    """Text-mode system prompt — used only when Google Search grounding is active."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%A, %B %d, %Y")
    return ARIA_GROUNDED_PROMPT_TEMPLATE.replace(
        "{current_date}", date_str
    ).replace(
        "{memory_context}", memory_context if memory_context else ""
    )


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
    # NEW: real-time search metadata — tells Android whether live data was used
    search_used: bool = False
    search_sources: Optional[list] = []   # list of {"title": ..., "url": ...}

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
    realtime_search_available: bool   # NEW
    version: str


# ─── AUTHENTICATION ───────────────────────────────────────────────────────────

def verify_token(request_token: str) -> bool:
    if not SECRET_TOKEN:
        logger.warning("No SECRET_TOKEN configured — auth disabled (dev mode)")
        return True
    return request_token == SECRET_TOKEN


# ─── GEMINI: STANDARD JSON MODE ───────────────────────────────────────────────

async def call_gemini(
    message: str,
    history: list,
    memory_context: str = "",
    screen_context: Optional[str] = None
) -> str:
    """
    Standard Gemini call — returns structured JSON.
    Used for all non-search queries.
    This is the existing call_gemini, unchanged in behavior.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")

    system_prompt = build_system_prompt(memory_context)

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
        # NOTE: No "tools" field here — adding google_search here would break
        # responseMimeType and return a 400. This is intentional.
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


# ─── GEMINI: GROUNDED SEARCH MODE ─────────────────────────────────────────────
#
# This is the NEW function. It activates Google Search grounding on Gemini.
#
# How Gemini grounding works:
#   1. You send a request with "tools": [{"google_search": {}}]
#   2. Gemini automatically decides what to search for
#   3. It searches Google, reads the results, and incorporates them
#   4. The response includes both the answer AND grounding metadata
#      (which URLs were consulted, which text was cited)
#   5. IMPORTANT: responseMimeType must NOT be "application/json" — grounding
#      returns enriched text, not JSON. We handle JSON wrapping ourselves.
#
# What we return:
#   A tuple: (aria_json_string, list_of_sources)
#   - aria_json_string: {"type":"chat","text":"..."} — standard ARIA format
#   - list_of_sources: [{"title":"...", "url":"..."}] — for transparency UI

async def call_gemini_with_grounding(
    message: str,
    history: list,
    memory_context: str = "",
    screen_context: Optional[str] = None
) -> tuple[str, list]:
    """
    Gemini call with Google Search grounding enabled.

    Returns: (aria_formatted_json, sources_list)

    The returned JSON is already in ARIA's standard format:
    {"type":"chat","text":"The weather in Addis Ababa is 22°C and sunny."}

    So ActionRouter on Android handles it exactly like a normal chat response.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")

    # Use the grounded prompt — plain text instructions, no JSON rules
    system_prompt = build_grounded_prompt(memory_context)

    final_message = message
    if screen_context:
        final_message = f"{message}\n\nScreen context:\n{screen_context[:1200]}"

    contents = list(history[-20:])
    contents.append({"role": "user", "parts": [{"text": final_message}]})

    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "tools": [
            {"google_search": {}}
            # This is the grounding tool declaration.
            # Gemini 2.5 Flash supports this natively.
            # It causes Gemini to automatically search Google when needed.
            # The search query is chosen by Gemini — we don't need to specify it.
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600
            # NOTE: NO responseMimeType here — incompatible with grounding.
            # Gemini will return plain text, which we wrap into JSON below.
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
        error_msg = error_data.get("error", {}).get("message", f"Gemini grounding error {response.status_code}")
        logger.error(f"Gemini grounding error: {error_msg}")
        # Fall through to normal call on grounding failure
        raise HTTPException(status_code=response.status_code, detail=error_msg)

    data = response.json()

    # ── Extract the text answer ────────────────────────────────────────────────
    candidate = data.get("candidates", [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    raw_text = ""
    for part in parts:
        if "text" in part:
            raw_text += part["text"]
    raw_text = raw_text.strip()

    # ── Extract grounding sources ──────────────────────────────────────────────
    #
    # Gemini grounding metadata lives at:
    # data["candidates"][0]["groundingMetadata"]["groundingChunks"]
    #
    # Each chunk has a "web" object with "uri" and "title".
    # We extract these and pass them back so the Android UI can
    # optionally show "Source: BBC News" under the response.
    #
    sources = []
    grounding_metadata = candidate.get("groundingMetadata", {})
    grounding_chunks = grounding_metadata.get("groundingChunks", [])
    for chunk in grounding_chunks:
        web = chunk.get("web", {})
        uri = web.get("uri", "")
        title = web.get("title", "")
        if uri and title:
            sources.append({"title": title, "url": uri})
        elif uri:
            sources.append({"title": uri, "url": uri})

    if sources:
        logger.info(f"Grounding returned {len(sources)} sources")
    else:
        logger.info("Grounding active but no source metadata returned")

    # ── Wrap the plain text answer into ARIA's standard JSON format ───────────
    #
    # Why this matters:
    # ActionRouter.kt on Android calls JSONObject(rawResponse) and checks
    # the "type" field. If we return plain text, it crashes with a JSONException.
    # By wrapping here, the Android code needs zero changes.
    #
    # We also append source attribution if sources were found — displayed
    # in the chat bubble as a small footnote.

    display_text = raw_text
    if sources:
        # Append a clean source line for the first source
        # (the most relevant one, which Gemini puts first)
        first = sources[0]
        display_text = f"{raw_text}\n\n📡 Source: {first['title']}"

    # Escape any quotes in the text so JSON stays valid
    safe_text = display_text.replace("\\", "\\\\").replace('"', '\\"').replace('\n', '\\n')
    aria_json = f'{{"type":"chat","text":"{safe_text}"}}'

    return aria_json, sources


# ─── OPENROUTER: STANDARD MODE ────────────────────────────────────────────────

async def call_openrouter(
    message: str,
    history: list,
    model: str,
    memory_context: str = "",
    screen_context: Optional[str] = None
) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured on server.")

    system_prompt = build_system_prompt(memory_context)

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


# ─── HELPERS ──────────────────────────────────────────────────────────────────

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
    search_used        = False      # NEW
    search_sources     = []         # NEW

    # ── Step 1: Retrieve relevant memories ────────────────────────────────────
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
                memory_context_str = memory_manager.format_for_context(relevant_memories)
                logger.info(f"Injected {memories_retrieved} memories into context.")
        except Exception as e:
            logger.warning(f"Memory retrieval failed (non-fatal): {e}")

    # ── Step 2: Decide routing path ────────────────────────────────────────────
    #
    # Priority order:
    #   A. Agent handles multi-step action requests (unchanged from v3.0.1)
    #   B. Grounded Gemini for real-time knowledge queries (NEW in v3.1.0)
    #   C. Normal Gemini/OpenRouter JSON call (unchanged from v3.0.1)
    #
    # The agent check runs FIRST because "search for weather today" should
    # still route through the device search tool if that's what ARIA decides.
    # The grounding check runs for everything else that has real-time signals.

    provider    = request.model or "gemini"
    ai_response = ""

    # ── Path A: Agent (multi-step actions) ────────────────────────────────────
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

    # ── Path B: Grounded search (real-time knowledge) ─────────────────────────
    if not agent_used and provider == "gemini" and needs_realtime_search(request.message):
        try:
            logger.info(f"Using Gemini grounding for: '{request.message[:60]}'")
            ai_response, search_sources = await call_gemini_with_grounding(
                message        = request.message,
                history        = request.history or [],
                memory_context = memory_context_str,
                screen_context = request.screen_context
            )
            search_used = True
            logger.info(f"Grounded response | sources={len(search_sources)}")

        except Exception as e:
            # Grounding failed — fall through to normal call
            # This can happen if the API key doesn't have grounding enabled
            # or if there's a quota issue. We degrade gracefully.
            logger.warning(f"Grounding failed, falling back to normal call: {e}")
            search_used = False
            ai_response = ""   # will be filled by Path C

    # ── Path C: Normal Gemini / OpenRouter call ────────────────────────────────
    if not agent_used and not search_used:
        message_with_memory = request.message
        if memory_context_str:
            message_with_memory = f"{memory_context_str}\n\nUser: {request.message}"

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

    # ── Step 3: Store new memories ─────────────────────────────────────────────
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
        except Exception as e:
            logger.warning(f"Memory storage failed (non-fatal): {e}")

    latency = int((time.time() - start_time) * 1000)
    logger.info(
        f"Response | agent={agent_used} | search={search_used} | "
        f"provider={provider} | latency={latency}ms | "
        f"mem_ret={memories_retrieved} | mem_stored={memories_stored} | "
        f"sources={len(search_sources)}"
    )

    return ChatResponse(
        response           = ai_response,
        provider           = provider,
        latency_ms         = latency,
        memories_retrieved = memories_retrieved,
        memories_stored    = memories_stored,
        agent_used         = agent_used,
        agent_tasks        = agent_tasks,
        success_rate       = success_rate,
        search_used        = search_used,
        search_sources     = search_sources
    )


# ─── MEMORY ENDPOINTS ─────────────────────────────────────────────────────────

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
        f'{{"type":"security","verdict":"safe|warning|danger","text":"your analysis in 2-3 sentences"}}\n\n'
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
        status                    = "online",
        gemini_configured         = bool(GEMINI_API_KEY),
        openrouter_configured     = bool(OPENROUTER_API_KEY),
        memory_available          = MEMORY_AVAILABLE,
        agent_enabled             = AGENT_AVAILABLE,
        realtime_search_available = bool(GEMINI_API_KEY),  # NEW
        version                   = "3.1.0"
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
    logger.info("=== ARIA Backend v3.1.0 starting up ===")
    logger.info(f"Gemini configured:          {bool(GEMINI_API_KEY)}")
    logger.info(f"OpenRouter configured:       {bool(OPENROUTER_API_KEY)}")
    logger.info(f"Auth token configured:       {bool(SECRET_TOKEN)}")
    logger.info(f"Memory system active:        {MEMORY_AVAILABLE}")
    logger.info(f"Agent system active:         {AGENT_AVAILABLE}")
    logger.info(f"Real-time search available:  {bool(GEMINI_API_KEY)}")
    logger.info("=========================================")

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