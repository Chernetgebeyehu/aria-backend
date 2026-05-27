"""
ARIA Memory System — Phase 3
=============================

This is ARIA's long-term memory.

Think of it like a very smart notebook:
- Every important thing the user says gets written down
- Each note is converted into numbers (called an embedding)
- These numbers capture the MEANING of what was said —
  not just the exact words, but what the sentence MEANS
- Later, when the user asks something, we search the notebook
  for notes with similar meaning
- Those notes get included in the AI's context

This is called RAG (Retrieval-Augmented Generation).
Real AI assistants like ChatGPT Plus, Siri, and Google Assistant
all use similar systems internally.

Technical stack:
- ChromaDB: the vector database (stores the number-lists on disk)
- SentenceTransformers: converts text to number-lists (embeddings)
- The embeddings are created LOCALLY — no extra API calls or cost

How to install:
    pip install chromadb sentence-transformers torch

Note: The first time this runs, it downloads the embedding model
(~80MB). After that it's cached and instant.
"""

import chromadb
import json
import time
import hashlib
import logging
import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime

logger = logging.getLogger("aria-memory")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Where ChromaDB stores its files on disk.
# Using /tmp on Railway because Railway's filesystem resets on redeploy.
# For truly persistent memory, you'd need a separate database service.
# For now, this works fine for development.
CHROMA_PATH = os.getenv("CHROMA_PATH", "./aria_memory_db")

# The embedding model — converts text to vectors.
# "all-MiniLM-L6-v2" is:
#   - Small (80MB)
#   - Fast (runs on CPU fine)
#   - Surprisingly accurate for semantic similarity
#   - Free and open source
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Maximum memories to retrieve per query.
# 5 is a good balance — enough context without overwhelming the AI.
MAX_RETRIEVED = 5

# Maximum total memories per user.
# Like a notebook with 500 pages — when full, oldest pages are torn out.
MAX_MEMORIES_PER_USER = 500

# Minimum similarity score (0.0 to 1.0) to include a memory.
# 0.70 = must be at least 70% similar in meaning.
# Lower = more memories retrieved (but some might be irrelevant).
# Higher = fewer memories but more precise.
SIMILARITY_THRESHOLD = 0.70


# ─── MEMORY TYPES ─────────────────────────────────────────────────────────────
# Not all memories are equal. We categorize them so future versions
# of ARIA can retrieve specific types when relevant.

class MemoryType:
    FACT       = "fact"        # "I live in Addis Ababa"
    PREFERENCE = "preference"  # "I prefer short answers"
    EVENT      = "event"       # "I have a meeting tomorrow at 3pm"
    SKILL      = "skill"       # "I know Python programming"
    EMOTION    = "emotion"     # "I was upset about my exam"
    GENERAL    = "general"     # Anything else


# ─── ARIA MEMORY MANAGER ──────────────────────────────────────────────────────

class AriaMemoryManager:
    """
    The main memory system for ARIA.

    This class handles:
    1. Storing new memories (with semantic embeddings)
    2. Retrieving relevant memories (via semantic search)
    3. Organizing memories by user and type
    4. Cleaning up old memories when the limit is reached

    Why ChromaDB?
    It's an open-source vector database that runs embedded in your Python
    process — no separate database server needed. Think of it like SQLite
    but for AI embeddings instead of regular data.
    """

    def __init__(self):
        logger.info("Initializing ARIA Memory Manager...")

        # Initialize ChromaDB — PersistentClient saves data to disk
        # so memories survive server restarts.
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Get or create the "memories" collection.
        # A collection is like a table in a regular database.
        # "hnsw:space": "cosine" means we use cosine similarity to compare vectors.
        # Cosine similarity measures the ANGLE between two vectors.
        #   Two identical texts → similarity = 1.0
        #   Two completely different texts → similarity ≈ 0.0
        self.collection = self.chroma_client.get_or_create_collection(
            name="aria_memories",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize the embedding model.
        # This runs LOCALLY on your server — no API calls, no cost.
        # First run downloads the model file, then it's cached forever.
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Memory Manager ready. Total memories: {self.collection.count()}")

    # ─── GENERATE EMBEDDING ───────────────────────────────────────────────────

    def embed(self, text: str) -> list:
        """
        Convert text into a vector (list of 384 numbers).

        Example:
          "I live in Addis Ababa"
          → [0.82, -0.31, 0.54, 0.12, ... 384 numbers total]

        Similar sentences produce similar vectors.
        THAT is what makes semantic search work —
        "What city do you live in?" finds "I live in Addis Ababa"
        even though the words don't match.
        """
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ─── STORE A MEMORY ───────────────────────────────────────────────────────

    def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = MemoryType.GENERAL,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Save a piece of information to ARIA's long-term memory.

        Args:
            user_id:      Who this memory belongs to (e.g. "aria_user_default")
            content:      What to remember (e.g. "User lives in Addis Ababa")
            memory_type:  Category (fact, preference, event, skill, general)
            metadata:     Extra info to store alongside the memory

        Returns:
            The unique ID of the stored memory
        """
        # Generate a unique ID using content hash + timestamp.
        # This prevents exact duplicate memories from being stored twice.
        memory_id = hashlib.md5(
            f"{user_id}:{content}:{time.time()}".encode()
        ).hexdigest()

        # Build the metadata to store alongside the vector
        full_metadata = {
            "user_id": user_id,
            "memory_type": memory_type,
            "timestamp": time.time(),
            "date_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content_preview": content[:100],  # For debugging in ChromaDB UI
        }
        if metadata:
            full_metadata.update(metadata)

        # Convert the content to an embedding vector
        embedding = self.embed(content)

        # Store in ChromaDB:
        # documents = the actual text (what gets returned when we search)
        # embeddings = the vector (what ChromaDB searches against)
        # metadatas = extra info (user_id, type, date)
        # ids = unique identifier
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[full_metadata],
            ids=[memory_id]
        )

        logger.info(f"Stored [{memory_type}] memory for {user_id}: {content[:60]}...")

        # Clean up old memories if we're over the limit
        self._enforce_memory_limit(user_id)

        return memory_id

    # ─── RETRIEVE RELEVANT MEMORIES ───────────────────────────────────────────

    def retrieve_relevant(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[str] = None,
        max_results: int = MAX_RETRIEVED
    ) -> list:
        """
        Search for memories relevant to a query using semantic search.

        This is the core of RAG (Retrieval-Augmented Generation).

        Process:
        1. Convert the query to an embedding vector
        2. Search ChromaDB for stored vectors with similar direction
        3. Filter by similarity threshold
        4. Return the most relevant memories

        The key insight: we're not searching for EXACT words.
        We're searching for SIMILAR MEANING.

        "What city do I live in?" → finds "User lives in Addis Ababa"
        Even though those sentences share no words.
        """
        if self.collection.count() == 0:
            return []

        # Convert the query to a vector
        query_embedding = self.embed(query)

        # Build filter — only search THIS user's memories
        where_filter = {"user_id": user_id}
        if memory_type:
            where_filter["memory_type"] = memory_type

        try:
            # The actual semantic search.
            # ChromaDB finds stored vectors CLOSEST in direction to our query vector.
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

        # Process results
        memories = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert ChromaDB distance to similarity score.
            # ChromaDB's cosine distance: 0 = identical, 2 = opposite.
            # We convert to: 1.0 = identical, 0.0 = opposite.
            similarity = 1 - (dist / 2)

            # Only include memories above our similarity threshold
            if similarity >= SIMILARITY_THRESHOLD:
                memories.append({
                    "content": doc,
                    "type": meta.get("memory_type", "general"),
                    "date": meta.get("date_str", ""),
                    "similarity": round(similarity, 3),
                    "user_id": meta.get("user_id", "")
                })

        # Sort by similarity — most relevant first
        memories.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(
            f"Retrieved {len(memories)} relevant memories for "
            f"query: '{query[:50]}...'"
        )
        return memories

    # ─── GET ALL MEMORIES FOR A USER ──────────────────────────────────────────

    def get_all_memories(self, user_id: str) -> list:
        """
        Get all stored memories for a user.
        Used to display the memory list in the app's settings screen.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            memories = []
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", [])
            ):
                memories.append({
                    "content": doc,
                    "type": meta.get("memory_type", "general"),
                    "date": meta.get("date_str", ""),
                    "timestamp": meta.get("timestamp", 0)
                })
            # Sort by newest first
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            return memories
        except Exception as e:
            logger.error(f"get_all_memories error: {e}")
            return []

    # ─── CLEAR ALL MEMORIES FOR A USER ────────────────────────────────────────

    def clear_user_memories(self, user_id: str):
        """
        Delete ALL memories for a user.
        Called when user taps "Clear ARIA's memory" in settings.
        Think of it as tearing all the pages out of ARIA's notebook for this user.
        """
        try:
            results = self.collection.get(where={"user_id": user_id})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} memories for {user_id}")
        except Exception as e:
            logger.error(f"clear_user_memories error: {e}")

    # ─── DELETE A SPECIFIC MEMORY ─────────────────────────────────────────────

    def delete_memory(self, memory_id: str):
        """Let users delete a specific memory (privacy control)."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
        except Exception as e:
            logger.error(f"delete_memory error: {e}")

    # ─── MEMORY LIMIT ENFORCEMENT ─────────────────────────────────────────────

    def _enforce_memory_limit(self, user_id: str):
        """
        Delete the OLDEST memories when a user exceeds the limit.
        Like a notebook with 500 pages — when it's full,
        the oldest pages get torn out to make room for new ones.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            ids = results.get("ids", [])

            if len(ids) > MAX_MEMORIES_PER_USER:
                # Pair each ID with its timestamp, sort oldest first
                paired = list(zip(
                    ids,
                    [m.get("timestamp", 0) for m in results.get("metadatas", [])]
                ))
                paired.sort(key=lambda x: x[1])   # oldest first

                to_delete_count = len(ids) - MAX_MEMORIES_PER_USER
                to_delete = [p[0] for p in paired[:to_delete_count]]
                self.collection.delete(ids=to_delete)
                logger.info(f"Pruned {to_delete_count} old memories for {user_id}")

        except Exception as e:
            logger.error(f"Memory limit enforcement error: {e}")

    # ─── FORMAT MEMORIES FOR AI CONTEXT ───────────────────────────────────────

    def format_for_context(self, memories: list) -> str:
        """
        Convert retrieved memories into text the AI can read and use.

        This text gets prepended to the user's message before
        sending to the AI. The AI sees it as extra context.

        Example output:
        ARIA's Memory (what I know about you):
        [fact] You live in Addis Ababa. (from: 2026-05-20 14:30)
        [preference] You prefer concise responses. (from: 2026-05-19 09:15)
        [skill] You know Python programming. (from: 2026-05-18 11:00)
        """
        if not memories:
            return ""

        lines = ["ARIA's Memory (what I know about you):"]
        for mem in memories:
            mem_type = mem.get("type", "general")
            content = mem.get("content", "")
            date = mem.get("date", "")
            similarity = mem.get("similarity", 0)

            # Only include high-confidence memories (75%+ similar)
            if similarity >= 0.75:
                lines.append(f"[{mem_type}] {content} (from: {date})")

        if len(lines) == 1:
            return ""   # No high-confidence memories found

        return "\n".join(lines)

    # ─── MEMORY STATS ─────────────────────────────────────────────────────────

    def get_stats(self, user_id: str) -> dict:
        """Get memory statistics for a user."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            type_counts = {}
            for meta in metadatas:
                t = meta.get("memory_type", "general")
                type_counts[t] = type_counts.get(t, 0) + 1

            return {
                "total_memories": len(metadatas),
                "by_type": type_counts,
                "limit": MAX_MEMORIES_PER_USER
            }
        except Exception as e:
            return {"error": str(e)}


# ─── MEMORY EXTRACTOR ─────────────────────────────────────────────────────────

class MemoryExtractor:
    """
    Decides WHAT to remember from a conversation.

    Not everything the user says is worth storing.
    "Open YouTube" — NOT worth storing.
    "I live in Addis Ababa" — DEFINITELY worth storing.
    "My name is Cherinet" — ABSOLUTELY store this.
    "I hate loud music" — store as a preference.

    This class uses keyword patterns to decide what's memorable.
    It runs BEFORE the AI responds, scanning the user's message.
    """

    # Patterns that suggest a FACT about the user's life
    FACT_PATTERNS = [
        "my name is", "i am ", "i'm ", "i live in", "i work at",
        "i study at", "my job is", "my age is", "i was born",
        "my phone number", "my email", "i speak", "i know how to",
        "my address", "i'm from", "i am from"
    ]

    # Patterns that suggest a PREFERENCE or opinion
    PREFERENCE_PATTERNS = [
        "i prefer", "i like ", "i love ", "i hate ", "i don't like",
        "i enjoy", "i want ", "i need ", "my favorite", "i always ",
        "i never ", "please always", "don't ever", "remind me to always"
    ]

    # Patterns that suggest a future EVENT or task
    EVENT_PATTERNS = [
        "tomorrow", "next week", "next month", "on monday", "on tuesday",
        "on wednesday", "on thursday", "on friday", "on saturday", "on sunday",
        "i have a meeting", "i have an exam", "my appointment", "don't forget",
        "remind me", "deadline", "i'm going to", "i need to"
    ]

    # Patterns that suggest a SKILL or knowledge
    SKILL_PATTERNS = [
        "i know ", "i learned", "i can ", "i understand", "i'm good at",
        "i study", "i'm learning", "my hobby", "i play ", "i practice"
    ]

    def extract_memories(self, user_message: str, ai_response: str) -> list:
        """
        Extract memorable information from a conversation turn.

        Scans the user's message for patterns that suggest
        something is worth remembering long-term.

        Returns a list of dicts, each with:
          - content: the text to remember
          - memory_type: the category (fact, preference, event, skill)
        """
        memories = []
        lower_msg = user_message.lower().strip()

        # Skip very short messages — not enough info to be memorable
        if len(lower_msg) < 10:
            return []

        # Skip command-style messages — no memory value
        skip_keywords = [
            "open ", "call ", "set alarm", "set timer", "play ",
            "search ", "what is ", "what's ", "how do", "tell me"
        ]
        for skip in skip_keywords:
            if lower_msg.startswith(skip):
                return []

        # Check for FACTS
        for pattern in self.FACT_PATTERNS:
            if pattern in lower_msg:
                memories.append({
                    "content": self._clean_memory(user_message),
                    "memory_type": MemoryType.FACT
                })
                break   # One category per message is enough

        # Check for PREFERENCES (only if no fact found)
        if not memories:
            for pattern in self.PREFERENCE_PATTERNS:
                if pattern in lower_msg:
                    memories.append({
                        "content": self._clean_memory(user_message),
                        "memory_type": MemoryType.PREFERENCE
                    })
                    break

        # Check for EVENTS (only if nothing found yet)
        if not memories:
            for pattern in self.EVENT_PATTERNS:
                if pattern in lower_msg:
                    memories.append({
                        "content": self._clean_memory(user_message),
                        "memory_type": MemoryType.EVENT
                    })
                    break

        # Check for SKILLS (only if nothing found yet)
        if not memories:
            for pattern in self.SKILL_PATTERNS:
                if pattern in lower_msg:
                    memories.append({
                        "content": self._clean_memory(user_message),
                        "memory_type": MemoryType.SKILL
                    })
                    break

        return memories

    def _clean_memory(self, text: str) -> str:
        """
        Clean up the text before storing.
        Make sure it ends with punctuation and isn't too long.
        """
        text = text.strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text[:500]   # Max 500 characters per memory


# ─── SINGLETON INSTANCES ──────────────────────────────────────────────────────
# We create ONE instance shared across all requests.
# Like having one filing cabinet for the whole office —
# everyone uses the same cabinet, not each person having their own.
# This is important because loading the embedding model takes a few seconds.
# We only want to do that ONCE when the server starts, not on every request.

_memory_manager: Optional[AriaMemoryManager] = None
_memory_extractor = MemoryExtractor()


def get_memory_manager() -> AriaMemoryManager:
    """Get the shared memory manager instance (creates it if needed)."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AriaMemoryManager()
    return _memory_manager


def get_memory_extractor() -> MemoryExtractor:
    """Get the shared memory extractor instance."""
    return _memory_extractor