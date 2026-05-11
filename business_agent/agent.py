
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import base64
import json
import logging
import sqlite3
from datetime import datetime, timezone
from neuronum import Cell
from model import get_model
from jinja2 import Environment, FileSystemLoader
import aiosqlite
import fitz


# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(filename="agent.log", level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

template_env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))

# ── Database ─────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.db")

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'finished',
            agent_name TEXT NOT NULL DEFAULT '',
            reference_id TEXT NOT NULL DEFAULT '',
            suggestions TEXT NOT NULL DEFAULT '',
            conversation TEXT NOT NULL DEFAULT ''
        )
    """)
    for col, default in [("status", "'finished'"), ("agent_name", "''"), ("reference_id", "''"), ("suggestions", "''"), ("conversation", "''")]:
        try:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
        except Exception:
            pass
    conn.commit()
    conn.close()

_init_db()

async def _append_conversation(task_id: int, new_messages: list, answer: str = None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT conversation FROM tasks WHERE id = ?", (task_id,)) as cur:
            row = await cur.fetchone()
        if row:
            try:
                existing = json.loads(row["conversation"] or "[]")
            except Exception:
                existing = []
            existing.extend(new_messages)
            if answer is not None:
                await db.execute("UPDATE tasks SET conversation = ?, answer = ? WHERE id = ?", (json.dumps(existing), answer, task_id))
            else:
                await db.execute("UPDATE tasks SET conversation = ? WHERE id = ?", (json.dumps(existing), task_id))
            await db.commit()

# ── Auth ─────────────────────────────────────────────────────────────────────

def is_authorized(sender: str, server_host: str, agent_id: str) -> bool:
    if agent_id:
        return False
    if sender == server_host:
        return True
    if "@" in sender and sender.split("@", 1)[1] == server_host:
        return True
    return False

def audience_allows(audience: str, sender: str, server_host: str = "") -> bool:
    """Returns True if sender is permitted by the agent's audience field."""
    if not audience:
        return False
    audience = audience.strip()
    if audience.lower() == "public":
        return True
    if audience.lower() == "private":
        return sender == server_host
    allowed = {s.strip() for s in audience.split(",")}
    if sender in allowed:
        return True
    sender_domain = sender.split("@", 1)[1] if "@" in sender else sender
    return sender_domain in allowed

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. "
    "Before giving your final answer, briefly explain your reasoning: what you considered, "
    "why you reached your conclusion, and any trade-offs or caveats worth noting. "
    "Structure your response as: reasoning first, then a clear answer."
)

# ── Handlers ─────────────────────────────────────────────────────────────────

async def route_to_agent(cell, query: str, sender: str):
    agents = await cell.list_agents() or []
    server_host = cell.host or cell.env.get("HOST", "")

    candidates = []
    for a in agents:
        try:
            config = json.loads(a.get("config", "{}"))
        except Exception:
            continue
        meta = config.get("agent_meta", {})
        audience = meta.get("audience", "")
        if not audience_allows(audience, sender, server_host):
            continue
        agent_id = meta.get("agent_id", a.get("agent_id", ""))
        creator = a.get("creator", "")
        author = a.get("author", "")
        verified = a.get("verified", "")
        for skill in config.get("skills", []):
            candidates.append({
                "agent_id": agent_id,
                "creator": creator,
                "author": author,
                "name": meta.get("name", ""),
                "logo": meta.get("logo", ""),
                "handle": skill.get("handle", ""),
                "description": skill.get("description", ""),
                "examples": skill.get("examples", []),
                "verified": verified,
            })
    print(candidates)
    if not candidates:
        return None

    skills_text = "\n".join(
        f'- agent_id="{c["agent_id"]}" handle="{c["handle"]}" description="{c["description"]}" examples={c["examples"]}'
        for c in candidates
    )

    system = (
        "You are a routing assistant. Given a user query and a list of available agent skills, "
        "choose up to 3 agents that are a strong, specific match for the query. "
        "If no agent is a strong match, return an empty array. "
        'Reply with ONLY a JSON array: [{"agent_id": "...", "handle": "..."}, ...]'
    )

    llm = get_model()
    result = await asyncio.to_thread(
        llm.create_chat_completion,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"User query: {query}\n\nAvailable skills:\n{skills_text}"},
        ]
    )
    raw = result["choices"][0]["message"]["content"].strip()

    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        choices = json.loads(raw[start:end])
        suggestions = []
        for choice in choices[:3]:
            agent_id = choice["agent_id"]
            handle = choice["handle"]
            matched = next((c for c in candidates if c["agent_id"] == agent_id and c["handle"] == handle), None)
            if matched:
                suggestions.append({
                    "agent_id": agent_id,
                    "creator": matched["creator"],
                    "author": matched.get("author", ""),
                    "handle": handle,
                    "agent_name": matched["name"],
                    "logo": matched.get("logo", ""),
                    "description": matched.get("description", ""),
                    "verified": matched.get("verified", 0),
                })
        return suggestions if suggestions else None
    except Exception:
        pass
    return None


async def handle_get_answer(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")
    query = data.get("query", "")
    context = data.get("context", "")

    llm = get_model()
    system = SYSTEM_PROMPT + (f"\n\nAdditional context:\n{context}" if context else "")
    llm_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    result = await asyncio.to_thread(llm.create_chat_completion, messages=llm_messages)
    answer = result["choices"][0]["message"]["content"]

    route = await route_to_agent(cell, query, cell_id)
    suggestions = [{**s, "query": query, "context": answer} for s in route] if route else []

    conversation = [{"role": "user", "text": query}, {"role": "agent", "text": answer}]
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (cell_id, query, answer, created_at, status, suggestions, conversation) VALUES (?, ?, ?, ?, 'open', ?, ?)",
            (cell_id, query, answer, created_at, json.dumps(suggestions), json.dumps(conversation))
        )
        task_id = cursor.lastrowid
        await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "suggestions": suggestions, "task_id": task_id}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_followup(cell, tx: dict):
    data = tx.get("data", {})
    task_id = data.get("task_id")
    query = data.get("query", "")
    context = data.get("context", "")

    history = []
    if task_id:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT conversation FROM tasks WHERE id = ?", (task_id,)) as cur:
                row = await cur.fetchone()
            if row:
                try:
                    history = json.loads(row["conversation"] or "[]")
                except Exception:
                    history = []

    llm = get_model()
    system = SYSTEM_PROMPT + (f"\n\nAdditional context:\n{context}" if context else "")
    messages = [{"role": "system", "content": system}]
    for m in history:
        role = "assistant" if m.get("role") == "agent" else "user"
        messages.append({"role": role, "content": m.get("text", "")})
    messages.append({"role": "user", "content": query})

    result = await asyncio.to_thread(llm.create_chat_completion, messages=messages)
    answer = result["choices"][0]["message"]["content"]

    if task_id:
        await _append_conversation(task_id, [{"role": "user", "text": query}, {"role": "agent", "text": answer}], answer)

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_read_file(cell, tx: dict):
    data = tx.get("data", {})
    filename = data.get("filename", "")
    content_b64 = data.get("content", "")

    if not content_b64:
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"context": ""}},
            client_public_key_str=data.get("public_key", "")
        )
        return

    ext = os.path.splitext(filename)[1].lower()
    try:
        raw = base64.b64decode(content_b64)
        if ext == ".pdf":
            doc = await asyncio.to_thread(fitz.open, stream=raw, filetype="pdf")
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        text = f"Could not read file: {e}"

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"context": text[:20000]}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_append_messages(cell, tx: dict):
    data = tx.get("data", {})
    task_id = data.get("task_id")
    new_messages = data.get("messages", [])

    if task_id and new_messages:
        await _append_conversation(task_id, new_messages)

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_finish_task(cell, tx: dict):
    data = tx.get("data", {})
    task_id = data.get("task_id")
    answer = data.get("answer", "")

    if task_id:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "UPDATE tasks SET status = 'finished', answer = CASE WHEN ? != '' THEN ? ELSE answer END WHERE id = ?",
                (answer, answer, task_id)
            )
            await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_load_tasks(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT query, answer, created_at, conversation FROM tasks WHERE cell_id = ? AND status = 'finished' ORDER BY created_at DESC LIMIT 100",
            (cell_id,)
        ) as cur:
            rows = await cur.fetchall()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"tasks": [dict(r) for r in rows]}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_load_open_tasks(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, query, answer, agent_name, created_at, suggestions, conversation FROM tasks WHERE cell_id = ? AND status = 'open' ORDER BY created_at DESC",
            (cell_id,)
        ) as cur:
            rows = await cur.fetchall()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"tasks": [dict(r) for r in rows]}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_ui(cell, tx: dict):
    data = tx.get("data", {})
    template = template_env.get_template("agent.html")
    host = cell.host or cell.env.get("HOST", "")
    operator = cell.env.get("OPERATOR", "")
    all_cells = await cell.list_cells(True)
    cells = [c for c in (all_cells or []) if "@" not in c.get("cell_id", "")]
    html = template.render(host=host, operator=operator, cells=cells)
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"html": html},
        client_public_key_str=data.get("public_key", "")
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

async def handle_tx(cell, tx: dict):
    try:
        data = tx.get("data", {})
        handle = data.get("handle", None)
        sender = tx.get("sender", "")
        server_host = cell.host or cell.env.get("HOST", "")
        agent_id = data.get("agent_id", None)

        if not is_authorized(sender, server_host, agent_id):
            logging.warning(f"Access denied: '{sender}' attempted '{handle}'")
            return

        handlers = {
            "get_answer": lambda: handle_get_answer(cell, tx),
            "get_followup": lambda: handle_get_followup(cell, tx),
            "read_file": lambda: handle_read_file(cell, tx),
            "append_messages": lambda: handle_append_messages(cell, tx),
            "finish_task": lambda: handle_finish_task(cell, tx),
            "load_tasks": lambda: handle_load_tasks(cell, tx),
            "load_open_tasks": lambda: handle_load_open_tasks(cell, tx),
            "get_ui": lambda: handle_get_ui(cell, tx),
        }

        handler = handlers.get(handle)
        if handler:
            await handler()

    except Exception as e:
        logging.error(f"Error: {e}")


async def start_agent(cell):
    async for tx in cell.sync():
        asyncio.create_task(handle_tx(cell, tx))


async def main():
    async with Cell() as cell:
        await start_agent(cell)


if __name__ == "__main__":
    asyncio.run(main())
