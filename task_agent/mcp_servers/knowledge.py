import os
import sqlite3
from datetime import datetime, timezone
from fastmcp import FastMCP
import aiosqlite

mcp = FastMCP("Knowledge Base")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.db")

MAX_CONTENT_LENGTH = 100_000
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
        USING fts5(title, content, tags, content=knowledge, content_rowid=id)
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
            INSERT INTO knowledge_fts(rowid, title, content, tags)
            VALUES (new.id, new.title, new.content, new.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
            INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, tags)
            VALUES ('delete', old.id, old.title, old.content, old.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
            INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, tags)
            VALUES ('delete', old.id, old.title, old.content, old.tags);
            INSERT INTO knowledge_fts(rowid, title, content, tags)
            VALUES (new.id, new.title, new.content, new.tags);
        END
    """)
    conn.commit()
    conn.close()

_init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_tags(tags: str) -> str:
    return ",".join(sorted(set(t.strip().lower() for t in tags.split(",") if t.strip())))

def _validate_limit(limit: int) -> int:
    return min(max(limit, 1), MAX_LIMIT)

def _validate_content(content: str):
    if len(content) > MAX_CONTENT_LENGTH:
        raise ValueError(f"Content exceeds maximum size of {MAX_CONTENT_LENGTH} characters.")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool
async def search_knowledge(query: str, limit: int = DEFAULT_LIMIT, offset: int = 0) -> dict:
    """Search knowledge entries using FTS5. Returns snippets — use get_entry() for full content."""
    limit = _validate_limit(limit)
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL;")
            async with db.execute("""
                SELECT k.id, k.title, k.tags, k.created_at, k.updated_at,
                       snippet(knowledge_fts, 1, '[[', ']]', '...', 24) AS snippet,
                       bm25(knowledge_fts) AS score
                FROM knowledge_fts
                JOIN knowledge k ON k.id = knowledge_fts.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY score
                LIMIT ? OFFSET ?
            """, (query, limit, offset)) as cur:
                rows = await cur.fetchall()
    except Exception as e:
        return {"success": False, "error": f"Invalid search query: {e}"}

    return {
        "success": True,
        "query": query,
        "count": len(rows),
        "results": [
            {
                "id": r["id"],
                "title": r["title"],
                "tags": r["tags"].split(",") if r["tags"] else [],
                "snippet": r["snippet"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "score": r["score"],
            }
            for r in rows
        ],
    }


@mcp.tool
async def get_entry(entry_id: int) -> dict:
    """Retrieve a single knowledge entry by ID with full content."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()

    if not row:
        return {"success": False, "error": f"No entry found with id {entry_id}"}

    return {
        "success": True,
        "entry": {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        },
    }


@mcp.tool
async def add_entry(title: str, content: str, tags: str = "") -> dict:
    """Add a new knowledge entry. Tags should be comma-separated."""
    _validate_content(content)
    now = _utc_now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        cursor = await db.execute(
            "INSERT INTO knowledge (title, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (title.strip(), content, _normalize_tags(tags), now, now)
        )
        await db.commit()
    return {"success": True, "entry_id": cursor.lastrowid}


@mcp.tool
async def update_entry(entry_id: int, title: str = None, content: str = None, tags: str = None) -> dict:
    """Update an existing knowledge entry. Only provided fields are updated."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL;")
        async with db.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"success": False, "error": f"No entry found with id {entry_id}"}

        final_content = content if content is not None else row["content"]
        _validate_content(final_content)

        await db.execute(
            "UPDATE knowledge SET title = ?, content = ?, tags = ?, updated_at = ? WHERE id = ?",
            (
                title.strip() if title is not None else row["title"],
                final_content,
                _normalize_tags(tags) if tags is not None else row["tags"],
                _utc_now(),
                entry_id,
            )
        )
        await db.commit()
    return {"success": True, "message": f"Entry {entry_id} updated."}


@mcp.tool
async def delete_entry(entry_id: int) -> dict:
    """Delete a knowledge entry by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        async with db.execute("SELECT id FROM knowledge WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"success": False, "error": f"No entry found with id {entry_id}"}
        await db.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
        await db.commit()
    return {"success": True, "message": f"Entry {entry_id} deleted."}


@mcp.tool
async def list_entries(limit: int = DEFAULT_LIMIT, offset: int = 0) -> dict:
    """List recent knowledge entries. Returns metadata only — use get_entry() for full content."""
    limit = _validate_limit(limit)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, title, tags, created_at, updated_at FROM knowledge ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ) as cur:
            rows = await cur.fetchall()

    return {
        "success": True,
        "count": len(rows),
        "results": [
            {
                "id": r["id"],
                "title": r["title"],
                "tags": r["tags"].split(",") if r["tags"] else [],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ],
    }


if __name__ == "__main__":
    mcp.run()
