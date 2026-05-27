"""
ARIA Agent System — Phase 4
============================

Before Phase 4:
    User message → AI → one action → done

After Phase 4:
    User message → Planner → Task list → Executor → each task
                → Verifier → success? → next task
                           → failed?  → Recovery → retry/fallback → continue
                → Final report to user

Think of it like a project manager who:
1. Gets a big request
2. Breaks it into small tasks
3. Assigns each task
4. Checks if each was done correctly
5. Handles problems without panicking
6. Reports back with full results

Real-world equivalent: AutoGPT, LangChain AgentExecutor, OpenAI Assistants API
all use this same loop pattern internally.
"""

import json
import time
import logging
import httpx
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("aria-agent")


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────
# These tell the AI exactly what tools exist and how to use them.
# This is called a "tool registry" — a catalogue of capabilities.
# The AI reads this and decides which tools to use for each task.

TOOL_REGISTRY = {
    "open_app": {
        "description": "Open/launch an installed app by name",
        "parameter": "app_name (string) — e.g. 'youtube', 'whatsapp', 'camera'",
        "example": {"tool": "open_app", "value": "youtube"},
        "category": "device"
    },
    "search": {
        "description": "Search the web or device for information",
        "parameter": "query (string) — the search query",
        "example": {"tool": "search", "value": "weather today Addis Ababa"},
        "category": "information"
    },
    "call": {
        "description": "Make a phone call to a contact or number",
        "parameter": "contact (string) — name or phone number",
        "example": {"tool": "call", "value": "mom"},
        "category": "communication"
    },
    "sms": {
        "description": "Send or open an SMS to a contact",
        "parameter": "contact (string) — name or phone number",
        "example": {"tool": "sms", "value": "dad"},
        "category": "communication"
    },
    "alarm": {
        "description": "Set a system alarm at a specific time",
        "parameter": "time (string) — e.g. '7:30am', '14:00', '9pm'",
        "example": {"tool": "alarm", "value": "7:30am"},
        "category": "time"
    },
    "timer": {
        "description": "Start a countdown timer for a duration",
        "parameter": "duration (string) — e.g. '5 minutes', '30 seconds'",
        "example": {"tool": "timer", "value": "10 minutes"},
        "category": "time"
    },
    "settings": {
        "description": "Open device settings",
        "parameter": "setting (string) — 'wifi', 'bluetooth', 'brightness', 'volume', 'general'",
        "example": {"tool": "settings", "value": "wifi"},
        "category": "device"
    },
    "open_url": {
        "description": "Open a specific URL in the browser",
        "parameter": "url (string) — full URL including https://",
        "example": {"tool": "open_url", "value": "https://google.com"},
        "category": "information"
    },
    "remember": {
        "description": "Store a fact or note in ARIA's long-term memory",
        "parameter": "content (string) — what to remember",
        "example": {"tool": "remember", "value": "User prefers concise responses"},
        "category": "memory"
    },
    "recall": {
        "description": "Search ARIA's long-term memory for information",
        "parameter": "query (string) — what to look for",
        "example": {"tool": "recall", "value": "user's name"},
        "category": "memory"
    }
}


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    SKIPPED   = "skipped"
    RECOVERED = "recovered"


@dataclass
class AgentTask:
    """
    One unit of work in the agent's plan.
    Like a single item on a to-do list, but smarter:
    it knows what to do if it fails, and it tracks its own status.
    """
    tool: str                              # Which tool to use (e.g. "call")
    value: str                             # The input for the tool (e.g. "mom")
    description: str                       # Human-readable description
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""                       # What happened when this was executed
    attempts: int = 0                      # How many times we tried
    fallback_tool: Optional[str] = None    # Alternative tool if this fails
    fallback_value: Optional[str] = None   # Alternative value if this fails
    critical: bool = False                 # If True, stop everything on failure


@dataclass
class AgentPlan:
    """
    The full plan for completing a user's request.
    Like a project plan with multiple tasks, a spoken summary, and a final report.
    """
    tasks: list = field(default_factory=list)
    spoken_intro: str = ""                  # What ARIA says before starting
    spoken_summary: str = ""               # What ARIA says after finishing
    original_request: str = ""             # The original user message
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """The final result after executing a full plan."""
    plan: AgentPlan
    succeeded: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    recovered: list = field(default_factory=list)
    total_time_ms: int = 0
    final_spoken: str = ""
    final_display: str = ""

    @property
    def success_rate(self) -> float:
        total = len(self.plan.tasks)
        if total == 0:
            return 1.0
        ok = len(self.succeeded) + len(self.recovered)
        return ok / total

    @property
    def fully_successful(self) -> bool:
        return len(self.failed) == 0


# ─── PLANNER ──────────────────────────────────────────────────────────────────

class AgentPlanner:
    """
    Turns a user's natural language request into a structured task plan.

    The planner sends the user's message to the AI with a special system prompt
    that forces the AI to output a JSON task plan instead of a normal response.

    This is called "structured output planning" — the same technique used by
    LangChain, AutoGPT, and Claude's tool-use system.
    """

    PLANNER_SYSTEM_PROMPT = (
        "You are ARIA's task planner. Your ONLY job is to convert user requests into executable task plans.\n\n"
        "You have access to these tools:\n"
        + json.dumps({k: v["description"] for k, v in TOOL_REGISTRY.items()}, indent=2)
        + """

CRITICAL RULES — READ EVERY LINE:

1. Respond ONLY with a raw JSON object. No markdown, no explanation, nothing else.

2. TASK SPLITTING — THIS IS THE MOST IMPORTANT RULE:
   When a message contains multiple actions connected by "and", "then", "also",
   you MUST create ONE SEPARATE task per action.
   NEVER combine two actions into a single task value.

   WRONG — DO NOT DO THIS:
   User: "call mom and send sms to dad"
   tasks: [{"tool": "call", "value": "mom and send sms to dad"}]  ← WRONG

   CORRECT — ALWAYS DO THIS:
   User: "call mom and send sms to dad"
   tasks: [
     {"tool": "call", "value": "mom"},
     {"tool": "sms",  "value": "dad"}
   ]

3. APP NAMES ARE SINGLE WORDS OR SHORT PHRASES — NOT FULL SENTENCES:
   The "value" for open_app is ONLY the app name, nothing else.
   WRONG: {"tool": "open_app", "value": "telegram and call mom"}  ← WRONG
   CORRECT: {"tool": "open_app", "value": "telegram"}             ← CORRECT

4. CONTACT NAMES ARE EXTRACTED CLEANLY:
   "call mom" → value is "mom" (not "mom and ..." or "mom, please")
   "send sms to dad" → tool: "sms", value: "dad"
   "message John" → tool: "sms", value: "John"

SCHEMA — every response must match this exactly:
{
  "needs_planning": true,
  "spoken_intro": "What ARIA says before starting (confident, natural, not robotic)",
  "tasks": [
    {
      "tool": "tool_name",
      "value": "SINGLE clean value — app name, contact name, time, etc.",
      "description": "human readable description of this one task",
      "fallback_tool": null,
      "fallback_value": null,
      "critical": false
    }
  ],
  "spoken_summary": "What ARIA says after completing everything"
}

If conversational (no action needed):
{"needs_planning": false, "response": "your reply here"}

═══ EXAMPLES — STUDY THESE ═══

User: "call mom and send sms to dad"
{"needs_planning":true,"spoken_intro":"On it! Calling mom and messaging dad.","tasks":[{"tool":"call","value":"mom","description":"Call mom","fallback_tool":null,"fallback_value":null,"critical":false},{"tool":"sms","value":"dad","description":"Send SMS to dad","fallback_tool":null,"fallback_value":null,"critical":false}],"spoken_summary":"Called mom and opened a message to dad."}

User: "open telegram and call mom"
{"needs_planning":true,"spoken_intro":"Opening Telegram and calling mom.","tasks":[{"tool":"open_app","value":"telegram","description":"Open Telegram","fallback_tool":"search","fallback_value":"telegram app","critical":false},{"tool":"call","value":"mom","description":"Call mom","fallback_tool":null,"fallback_value":null,"critical":false}],"spoken_summary":"Opened Telegram and called mom."}

User: "set alarm for 8am and open youtube"
{"needs_planning":true,"spoken_intro":"Setting your alarm and opening YouTube!","tasks":[{"tool":"alarm","value":"8am","description":"Set alarm for 8am","fallback_tool":null,"fallback_value":null,"critical":false},{"tool":"open_app","value":"youtube","description":"Open YouTube","fallback_tool":"search","fallback_value":"youtube","critical":false}],"spoken_summary":"Alarm set for 8am and YouTube is opening."}

User: "send a message to John then call Sarah"
{"needs_planning":true,"spoken_intro":"Sure! Messaging John and then calling Sarah.","tasks":[{"tool":"sms","value":"John","description":"Send SMS to John","fallback_tool":null,"fallback_value":null,"critical":false},{"tool":"call","value":"Sarah","description":"Call Sarah","fallback_tool":null,"fallback_value":null,"critical":false}],"spoken_summary":"Opened a message to John and called Sarah."}

User: "open whatsapp and search for flights to Dubai"
{"needs_planning":true,"spoken_intro":"Opening WhatsApp and searching for flights.","tasks":[{"tool":"open_app","value":"whatsapp","description":"Open WhatsApp","fallback_tool":"search","fallback_value":"whatsapp","critical":false},{"tool":"search","value":"flights to Dubai","description":"Search for flights to Dubai","fallback_tool":null,"fallback_value":null,"critical":false}],"spoken_summary":"WhatsApp is open and searching for flights to Dubai."}

User: "turn on wifi"
{"needs_planning":true,"spoken_intro":"Opening WiFi settings.","tasks":[{"tool":"settings","value":"wifi","description":"Open WiFi settings","fallback_tool":null,"fallback_value":null,"critical":false}],"spoken_summary":"WiFi settings opened."}

User: "what is the weather like today?"
{"needs_planning":false,"response":"I can search that for you! Would you like me to search the current weather?"}"""
    )

    def __init__(self, gemini_api_key: str, openrouter_api_key: str = ""):
        self.gemini_key = gemini_api_key
        self.openrouter_key = openrouter_api_key
        self.gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent"
        )

    async def create_plan(
        self,
        user_message: str,
        memory_context: str = "",
        provider: str = "gemini"
    ) -> AgentPlan:
        """Send the user's message to the AI and get back a structured task plan."""
        augmented = user_message
        if memory_context:
            augmented = f"{user_message}\n\n[Memory context: {memory_context}]"

        raw_json = await self._call_planner_ai(augmented, provider)
        return self._parse_plan(raw_json, user_message)

    async def _call_planner_ai(self, message: str, provider: str) -> str:
        if provider == "gemini" and self.gemini_key:
            return await self._gemini_plan(message)
        elif self.openrouter_key:
            return await self._openrouter_plan(message)
        else:
            raise ValueError("No AI provider configured for planning")

    async def _gemini_plan(self, message: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": self.PLANNER_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": message}]}],
            "generationConfig": {
                "temperature": 0.2,       # Low temperature = more structured output
                "maxOutputTokens": 800,
                "responseMimeType": "application/json"
            }
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.gemini_url}?key={self.gemini_key}",
                json=body
            )
        if not response.is_success:
            raise Exception(f"Gemini planner error: {response.status_code}")
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _openrouter_plan(self, message: str) -> str:
        body = {
            "model": "google/gemma-3-4b-it:free",
            "messages": [
                {"role": "system", "content": self.PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                }
            )
        if not response.is_success:
            raise Exception(f"OpenRouter planner error: {response.status_code}")
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _parse_plan(self, raw_json: str, original_request: str) -> AgentPlan:
        """
        Parse the AI's JSON response into an AgentPlan.
        Handles malformed JSON gracefully.
        """
        text = raw_json.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```")
            end = text.rfind("```")
            if end >= 0:
                text = text[:end]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Plan parsing failed: {e}\nRaw: {text[:200]}")
            # Return a fallback that just does normal chat
            return AgentPlan(
                tasks=[],
                spoken_intro="",
                spoken_summary=raw_json,
                original_request=original_request
            )

        # If the AI decided no planning is needed (conversational)
        if not data.get("needs_planning", True):
            return AgentPlan(
                tasks=[],
                spoken_intro="",
                spoken_summary=data.get("response", ""),
                original_request=original_request
            )

        # Parse task list
        tasks = []
        for t in data.get("tasks", []):
            tool = t.get("tool", "")
            if tool not in TOOL_REGISTRY:
                logger.warning(f"Unknown tool in plan: {tool} — skipping")
                continue
            tasks.append(AgentTask(
                tool=tool,
                value=t.get("value", ""),
                description=t.get("description", f"{tool}: {t.get('value', '')}"),
                fallback_tool=t.get("fallback_tool"),
                fallback_value=t.get("fallback_value"),
                critical=t.get("critical", False)
            ))

        return AgentPlan(
            tasks=tasks,
            spoken_intro=data.get("spoken_intro", ""),
            spoken_summary=data.get("spoken_summary", ""),
            original_request=original_request
        )


# ─── EXECUTOR ─────────────────────────────────────────────────────────────────

class AgentExecutor:
    """
    Executes an AgentPlan step by step.

    For each task:
    1. Try the primary tool
    2. If it fails, try the fallback
    3. If fallback also fails and task is critical → stop
    4. Otherwise → mark as failed and continue

    This is called "graceful degradation" — the agent does as much as possible
    even when some things fail. Like a surgeon who hits a complication but
    stabilizes the patient instead of just walking away.
    """

    MAX_RETRIES = 2
    RETRY_DELAY = 1.0  # seconds between retries

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    async def execute(self, plan: AgentPlan, user_id: str = "default") -> AgentResult:
        """Execute all tasks in the plan sequentially."""
        start_time = time.time()
        result = AgentResult(plan=plan)

        if not plan.tasks:
            # No tasks — this was a conversational response
            result.final_spoken = plan.spoken_summary
            result.final_display = plan.spoken_summary
            result.total_time_ms = int((time.time() - start_time) * 1000)
            return result

        logger.info(f"Executing plan with {len(plan.tasks)} tasks for user {user_id}")

        for i, task in enumerate(plan.tasks):
            logger.info(f"Task {i+1}/{len(plan.tasks)}: {task.tool}({task.value})")
            task.status = TaskStatus.RUNNING

            success = await self._execute_task(task, user_id)

            if success:
                result.succeeded.append(task)
            else:
                # Try fallback if available
                if task.fallback_tool:
                    logger.info(
                        f"Primary failed, trying fallback: "
                        f"{task.fallback_tool}({task.fallback_value})"
                    )
                    fallback_success = await self._execute_fallback(task, user_id)
                    if fallback_success:
                        task.status = TaskStatus.RECOVERED
                        result.recovered.append(task)
                        continue

                # Task fully failed
                task.status = TaskStatus.FAILED
                result.failed.append(task)

                # If critical, abort the plan
                if task.critical:
                    logger.warning(f"Critical task failed: {task.tool}. Aborting plan.")
                    for remaining in plan.tasks[i + 1:]:
                        remaining.status = TaskStatus.SKIPPED
                    break

        result.total_time_ms = int((time.time() - start_time) * 1000)
        result.final_spoken = self._build_spoken_summary(result, plan)
        result.final_display = self._build_display_summary(result)

        logger.info(
            f"Plan complete: {len(result.succeeded)} succeeded, "
            f"{len(result.recovered)} recovered, {len(result.failed)} failed "
            f"in {result.total_time_ms}ms"
        )
        return result

    async def _execute_task(self, task: AgentTask, user_id: str) -> bool:
        task.attempts += 1
        try:
            result_text = await self._run_tool(task.tool, task.value, user_id)
            task.result = result_text
            task.status = TaskStatus.SUCCESS
            await asyncio.sleep(0.3)  # Small delay between tasks
            return True
        except Exception as e:
            logger.error(f"Task {task.tool}({task.value}) failed: {e}")
            task.result = str(e)
            if task.attempts < self.MAX_RETRIES:
                logger.info(f"Retrying task (attempt {task.attempts + 1})")
                await asyncio.sleep(self.RETRY_DELAY)
                return await self._execute_task(task, user_id)
            return False

    async def _execute_fallback(self, task: AgentTask, user_id: str) -> bool:
        if not task.fallback_tool:
            return False
        try:
            result_text = await self._run_tool(
                task.fallback_tool,
                task.fallback_value or task.value,
                user_id
            )
            task.result = f"[via fallback {task.fallback_tool}] {result_text}"
            return True
        except Exception as e:
            logger.error(f"Fallback {task.fallback_tool} also failed: {e}")
            task.result = f"Both {task.tool} and {task.fallback_tool} failed"
            return False

    async def _run_tool(self, tool: str, value: str, user_id: str) -> str:
        """
        Execute a specific tool.

        Memory tools (remember/recall) run server-side here.
        All other tools are device-side — the server validates and queues them,
        then the Android ActionRouter executes them when it receives the response.
        """
        if tool == "remember" and self.memory_manager:
            self.memory_manager.store_memory(
                user_id=user_id,
                content=value,
                memory_type="fact"
            )
            return f"Remembered: {value}"

        elif tool == "recall" and self.memory_manager:
            memories = self.memory_manager.retrieve_relevant(
                user_id=user_id,
                query=value,
                max_results=3
            )
            if memories:
                return " | ".join(m["content"] for m in memories)
            return "Nothing found in memory for that query."

        else:
            # All other tools are device-side — validate and pass through
            if tool not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool: {tool}")
            if not value and tool not in ("settings",):
                raise ValueError(f"Tool {tool} requires a value")
            # Return confirmation — actual execution happens on Android
            return f"Queued: {tool}({value})"

    def _build_spoken_summary(self, result: AgentResult, plan: AgentPlan) -> str:
        if result.fully_successful:
            if plan.spoken_summary:
                return plan.spoken_summary
            count = len(result.succeeded) + len(result.recovered)
            return f"Done! I completed all {count} tasks."
        elif result.success_rate >= 0.5:
            ok = len(result.succeeded) + len(result.recovered)
            fail = len(result.failed)
            return (
                f"I completed {ok} out of {ok + fail} tasks. "
                f"{fail} couldn't be done right now."
            )
        else:
            return "I ran into some issues completing your request. Please try again."

    def _build_display_summary(self, result: AgentResult) -> str:
        lines = []
        for task in result.plan.tasks:
            if task.status == TaskStatus.SUCCESS:
                lines.append(f"✅ {task.description}")
            elif task.status == TaskStatus.RECOVERED:
                lines.append(f"✅ {task.description} (via fallback)")
            elif task.status == TaskStatus.FAILED:
                lines.append(f"❌ {task.description}")
            elif task.status == TaskStatus.SKIPPED:
                lines.append(f"⏭️ {task.description} (skipped)")
        if result.recovered:
            lines.append(f"\n⚡ {len(result.recovered)} task(s) used fallback strategies")
        if result.plan.tasks:
            lines.append(f"\n⏱️ Completed in {result.total_time_ms}ms")
        return "\n".join(lines)


# ─── ORCHESTRATOR ─────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    The top-level coordinator that ties Planner + Executor together.
    This is what main.py calls.

    Flow:
    1. Check if this request needs agent planning (simple requests skip for speed)
    2. If yes: Planner → Plan → Executor → Result
    3. If no: Signal main.py to use normal chat

    Deciding when to use the agent vs normal chat:
    - Multi-step requests → agent
    - Questions → normal chat
    - Single actions → agent (still more reliable for device actions)
    - Greetings/small talk → normal chat
    """

    # Keywords that suggest multi-step actions — checked after action verbs
    MULTI_ACTION_KEYWORDS = [
        " and ", " then ", " also ", " after that ", " plus ",
        "first ", "next ", "finally ", "lastly ", "as well"
    ]

    def __init__(
        self,
        gemini_api_key: str,
        openrouter_api_key: str = "",
        memory_manager=None
    ):
        self.planner = AgentPlanner(gemini_api_key, openrouter_api_key)
        self.executor = AgentExecutor(memory_manager)

    def needs_agent(self, message: str) -> bool:
        """
        Decide if a message needs the full agent loop or just a simple chat.

        The agent loop adds ~500-800ms latency vs direct chat,
        so we skip it when the request is clearly conversational.

        IMPORTANT: Action signals take priority over conversation signals.
        "call mom and what is the weather?" contains "?" (conversation signal)
        but also "call " (action signal) — it should go through the agent.
        We check action signals FIRST, then conversation signals.
        """
        lower = message.lower().strip()

        # ── Step 1: Check for action verbs FIRST (higher priority) ────────────
        # These indicate the user wants ARIA to DO something on the device.
        # If any of these match, always use the agent regardless of other signals.
        action_verbs = [
            "open ", "call ", "send ", "set ", "play ", "start ",
            "turn on", "turn off", "search for", "message ",
            "ring ", "dial ", "launch ", "show me", "remind me",
        ]
        for verb in action_verbs:
            if lower.startswith(verb) or f" {verb}" in lower:
                return True

        # ── Step 2: Check for multi-step connectors ────────────────────────────
        # These explicitly chain multiple actions together.
        for kw in self.MULTI_ACTION_KEYWORDS:
            if kw in lower:
                return True

        # ── Step 3: Pure conversation signals → skip agent for speed ──────────
        # Only reach here if no action verbs were found above.
        # "?" alone doesn't mean no action — but combined with no action verb
        # it means it's a question ARIA should just answer.
        conversation_only = [
            "what is", "what are", "who is", "how does", "explain",
            "tell me about", "why does", "what do you think",
            "help me understand", "what's the difference",
            "what time is it", "how are you", "what's up",
        ]
        for kw in conversation_only:
            if lower.startswith(kw) or lower == kw.strip():
                return False

        # ── Step 4: Default → use agent ───────────────────────────────────────
        # If we're not sure, the agent handles it safely.
        # The planner will return needs_planning=false for pure chat messages
        # and we fall back to direct chat in that case anyway.
        return True

    async def run(
        self,
        user_message: str,
        memory_context: str = "",
        provider: str = "gemini",
        user_id: str = "default"
    ) -> dict:
        """
        Main entry point. Returns a dict compatible with ARIA's existing
        response format so the Android app doesn't need major changes.
        """
        use_agent = self.needs_agent(user_message)

        logger.info(
            f"Message: '{user_message[:60]}' | "
            f"Agent: {use_agent} | Provider: {provider}"
        )

        if not use_agent:
            # Signal to main.py to use the normal chat endpoint
            return {"use_agent": False}

        # ── Agent flow ─────────────────────────────────────────────────────────
        try:
            # Step 1: Plan
            plan = await self.planner.create_plan(
                user_message=user_message,
                memory_context=memory_context,
                provider=provider
            )

            # Step 2: If no tasks, it's conversational
            if not plan.tasks:
                return {
                    "use_agent": True,
                    "response": json.dumps({
                        "type": "chat",
                        "text": plan.spoken_summary
                    }),
                    "tasks": [],
                    "success_rate": 1.0
                }

            # Step 3: Execute
            result = await self.executor.execute(plan, user_id)

            # Step 4: Build ARIA-compatible response
            # Include the task list so Android ActionRouter can execute device-side tools
            return {
                "use_agent": True,
                "response": json.dumps({
                    "type": "agent_result",
                    "spoken": result.final_spoken,
                    "display": result.final_display,
                    "tasks": [
                        {
                            "tool": t.tool,
                            "value": t.value,
                            "status": t.status.value
                        }
                        for t in plan.tasks
                    ],
                    "success_rate": result.success_rate
                }),
                "tasks": [
                    {
                        "tool": t.tool,
                        "value": t.value,
                        "status": t.status.value,
                        "result": t.result
                    }
                    for t in plan.tasks
                ],
                "success_rate": result.success_rate,
                "spoken": result.final_spoken,
                "display": result.final_display
            }

        except Exception as e:
            logger.error(f"Agent orchestration failed: {e}", exc_info=True)
            # Fall back to normal chat on agent failure
            return {"use_agent": False, "agent_error": str(e)}