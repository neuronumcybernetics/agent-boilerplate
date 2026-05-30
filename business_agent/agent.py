
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            agents TEXT NOT NULL DEFAULT '[]',
            employees TEXT NOT NULL DEFAULT '[]',
            knowledge TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    try:
        conn.execute("ALTER TABLE task_templates ADD COLUMN knowledge TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE task_templates ADD COLUMN mode TEXT NOT NULL DEFAULT 'parallel'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE task_templates ADD COLUMN parameters TEXT NOT NULL DEFAULT '[]'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE task_templates ADD COLUMN last_used_at TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '',
            cells TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
    """)
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

def is_authorized(sender: str, server_host: str, agent_id: str) -> tuple[bool, str]:
    """Returns (authorized, access_level) where access_level is 'admin', 'employee', or ''."""
    if agent_id:
        return False, ""
    if sender == server_host:
        return True, "admin"
    if "@" in sender and sender.split("@", 1)[1] == server_host:
        return True, "employee"
    return False, ""


# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful personal assistant. "
    "Before giving your final answer, briefly explain your reasoning: what you considered, "
    "why you reached your conclusion, and any trade-offs or caveats worth noting. "
    "Structure your response as: reasoning first, then a clear answer."
)

HELP_PROMPT = (
    "You are a helpful assistant embedded in a business agent workspace. "
    "Your job is to answer questions about how to use the workspace. "
    "The workspace has the following features:\n"
    "- Task Templates: pre-configured workflows that route tasks to one or more AI agents. Admins create them, employees use them.\n"
    "- Agents: external AI agents (e.g. CRM Agent, Email Agent, ERP Agent) that handle specialized tasks.\n"
    "- Knowledge Base: a searchable store of documents and notes used as context by the local AI.\n"
    "- Tasks: conversations and results are saved as tasks (open or finished) and searchable.\n"
    "- Step-by-step mode: templates can chain agents sequentially, passing each agent's output as context to the next.\n"
    "- Parallel mode: templates can run multiple agents simultaneously and combine their answers.\n"
    "Be concise, friendly, and practical. If the user asks something unrelated to the workspace, politely redirect them."
)

# ── Handlers ─────────────────────────────────────────────────────────────────


async def _load_knowledge_for(cell_id: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT title, content, tags, cells FROM knowledge ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    relevant = []
    for r in rows:
        cells = json.loads(r["cells"] or "[]")
        if not cells or cell_id in cells:
            relevant.append(f"### {r['title']}\n{r['content']}")
    return "\n\n".join(relevant)


async def handle_get_help(cell, tx: dict):
    data = tx.get("data", {})
    query = data.get("query", "")
    llm = get_model()
    result = await asyncio.to_thread(llm.create_chat_completion, messages=[
        {"role": "system", "content": HELP_PROMPT},
        {"role": "user", "content": query},
    ])
    answer = result["choices"][0]["message"]["content"]
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_answer(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")
    query = data.get("query", "")
    context = data.get("context", "")

    knowledge = await _load_knowledge_for(cell_id)

    system = SYSTEM_PROMPT
    extras = []
    if knowledge:
        extras.append(f"Knowledge base:\n{knowledge}")
    if context:
        extras.append(f"Additional context:\n{context}")
    if extras:
        system += "\n\n" + "\n\n".join(extras)

    llm = get_model()
    result = await asyncio.to_thread(llm.create_chat_completion, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ])
    answer = result["choices"][0]["message"]["content"]

    conversation = [{"role": "user", "text": query}, {"role": "agent", "text": answer}]
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (cell_id, query, answer, created_at, status, suggestions, conversation) VALUES (?, ?, ?, ?, 'open', '[]', ?)",
            (cell_id, query, answer, created_at, json.dumps(conversation))
        )
        task_id = cursor.lastrowid
        await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "task_id": task_id}},
        client_public_key_str=data.get("public_key", "")
    )


async def _call_agent(cell, agent_id: str, creator: str, handle: str, query: str, context: str) -> str:
    try:
        result = await cell.activate_tx(
            data={"handle": handle, "agent_id": agent_id, "query": query, "context": context},
            cell_id=creator
        )
        json_result = result.get("data", {}).get("json", result.get("json", {}))
        return json_result.get("answer", str(result))
    except Exception as e:
        return f"Error contacting agent: {e}"


async def handle_template_task(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")
    query = data.get("query", "")
    knowledge = data.get("knowledge", "")
    context = f"Task knowledge:\n{knowledge}" if knowledge else ""

    # Support single agent (legacy) or list of agents
    agents = data.get("agents", [])
    if not agents:
        agent_id = data.get("target_agent_id", "")
        creator = data.get("creator", "")
        handle = data.get("target_handle", "")
        if agent_id and creator:
            agents = [{"agent_id": agent_id, "creator": creator, "handle": handle}]

    if not agents:
        await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"answer": "No agents specified.", "task_id": None}}, client_public_key_str=data.get("public_key", ""))
        return

    mode = data.get("mode", "parallel")

    if mode == "sequential":
        steps = []
        step_context = context
        for a in agents:
            name = a.get("name", a["agent_id"])
            result = await _call_agent(cell, a["agent_id"], a["creator"], a.get("handle", ""), query, step_context)
            steps.append(f"**{name}:** {result}")
            step_context = (context + "\n\n" if context else "") + f"Previous step ({name}):\n{result}"
        answer = "\n\n".join(steps) if len(steps) > 1 else steps[0]
    else:
        results = await asyncio.gather(*[
            _call_agent(cell, a["agent_id"], a["creator"], a.get("handle", ""), query, context)
            for a in agents
        ])
        agent_names = [a.get("name", a["agent_id"]) for a in agents]
        answer = results[0] if len(agents) == 1 else "\n\n".join(f"**{name}:** {r}" for name, r in zip(agent_names, results))

    conversation = [{"role": "user", "text": query}, {"role": "agent", "text": answer}]
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (cell_id, query, answer, created_at, status, suggestions, conversation) VALUES (?, ?, ?, ?, 'open', '[]', ?)",
            (cell_id, query, answer, created_at, json.dumps(conversation))
        )
        task_id = cursor.lastrowid
        await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "task_id": task_id}},
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


async def handle_create_task_template(cell, tx: dict):
    data = tx.get("data", {})
    name = data.get("name", "").strip()
    if not name:
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"ok": False, "error": "name is required"}},
            client_public_key_str=data.get("public_key", "")
        )
        return
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO task_templates (name, description, agents, employees, knowledge, mode, parameters, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, data.get("description", ""), json.dumps(data.get("agents", [])), json.dumps(data.get("employees", [])), data.get("knowledge", ""), data.get("mode", "parallel"), json.dumps(data.get("parameters", [])), created_at)
        )
        template_id = cursor.lastrowid
        await db.commit()
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True, "template_id": template_id}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_load_task_templates(cell, tx: dict):
    data = tx.get("data", {})
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, description, agents, employees, knowledge, mode, parameters, created_at, last_used_at FROM task_templates ORDER BY CASE WHEN last_used_at != '' THEN last_used_at ELSE created_at END DESC"
        ) as cur:
            rows = await cur.fetchall()
    templates = [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "agents": json.loads(r["agents"] or "[]"),
            "employees": json.loads(r["employees"] or "[]"),
            "knowledge": r["knowledge"] or "",
            "mode": r["mode"] or "parallel",
            "parameters": json.loads(r["parameters"] or "[]"),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"templates": templates}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_use_task_template(cell, tx: dict):
    data = tx.get("data", {})
    template_id = data.get("template_id")
    if template_id:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE task_templates SET last_used_at = ? WHERE id = ?", (datetime.now(timezone.utc).isoformat(), template_id))
            await db.commit()
    await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": True}}, client_public_key_str=data.get("public_key", ""))


async def handle_delete_task_template(cell, tx: dict):
    data = tx.get("data", {})
    template_id = data.get("template_id")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM task_templates WHERE id = ?", (template_id,))
        await db.commit()
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_upload_knowledge(cell, tx: dict):
    data = tx.get("data", {})
    filename = data.get("filename", "")
    content_b64 = data.get("content", "")
    cells = data.get("cells", [])

    if not content_b64:
        await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": False, "error": "no content"}}, client_public_key_str=data.get("public_key", ""))
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
        await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": False, "error": str(e)}}, client_public_key_str=data.get("public_key", ""))
        return

    title = os.path.splitext(filename)[0]
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO knowledge (title, content, tags, cells, created_at) VALUES (?, ?, ?, ?, ?)",
            (title, text[:20000], ext.lstrip("."), json.dumps(cells), created_at)
        )
        entry_id = cursor.lastrowid
        await db.commit()

    await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": True, "entry_id": entry_id, "title": title}}, client_public_key_str=data.get("public_key", ""))


async def handle_add_knowledge(cell, tx: dict):
    data = tx.get("data", {})
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    if not title or not content:
        await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": False, "error": "title and content required"}}, client_public_key_str=data.get("public_key", ""))
        return
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO knowledge (title, content, tags, cells, created_at) VALUES (?, ?, ?, ?, ?)",
            (title, content, data.get("tags", ""), json.dumps(data.get("cells", [])), created_at)
        )
        entry_id = cursor.lastrowid
        await db.commit()
    await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": True, "entry_id": entry_id}}, client_public_key_str=data.get("public_key", ""))


async def handle_delete_knowledge(cell, tx: dict):
    data = tx.get("data", {})
    entry_id = data.get("entry_id")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
        await db.commit()
    await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"ok": True}}, client_public_key_str=data.get("public_key", ""))


async def handle_load_knowledge(cell, tx: dict):
    data = tx.get("data", {})
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id, title, content, tags, cells, created_at FROM knowledge ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    entries = [{"id": r["id"], "title": r["title"], "content": r["content"], "tags": r["tags"], "cells": json.loads(r["cells"] or "[]"), "created_at": r["created_at"]} for r in rows]
    await cell.tx_response(tx_id=tx.get("tx_id"), data={"json": {"entries": entries}}, client_public_key_str=data.get("public_key", ""))


async def handle_get_ui(cell, tx: dict, access_level: str = "employee"):
    data = tx.get("data", {})
    template = template_env.get_template("agent.html")
    host = cell.host or cell.env.get("HOST", "")
    operator = cell.env.get("OPERATOR", "")
    all_cells = await cell.list_cells(True) or []
    cells = [c for c in all_cells if "@" not in c.get("cell_id", "")]
    employees = [c for c in all_cells if "@" in c.get("cell_id", "")]
    raw_agents = await cell.list_agents() or []
    agents = []
    for a in raw_agents:
        try:
            config = json.loads(a.get("config", "{}"))
        except Exception:
            config = {}
        meta = config.get("agent_meta", {})
        agents.append({
            "agent_id": meta.get("agent_id", a.get("agent_id", "")),
            "name": meta.get("name", a.get("agent_id", "")),
            "logo": meta.get("logo", ""),
            "description": meta.get("description", ""),
            "creator": a.get("creator", ""),
            "author": a.get("author", ""),
            "verified": bool(a.get("verified", 0)),
        })
    html = template.render(host=host, operator=operator, cells=cells, employees=employees, agents=agents, access_level=access_level)
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

        authorized, access_level = is_authorized(sender, server_host, agent_id)
        if not authorized:
            logging.warning(f"Access denied: '{sender}' attempted '{handle}'")
            return

        handlers = {
            "get_answer": lambda: handle_get_answer(cell, tx),
            "get_help": lambda: handle_get_help(cell, tx),
            "get_followup": lambda: handle_get_followup(cell, tx),
            "template_task": lambda: handle_template_task(cell, tx),
            "read_file": lambda: handle_read_file(cell, tx),
            "append_messages": lambda: handle_append_messages(cell, tx),
            "finish_task": lambda: handle_finish_task(cell, tx),
            "load_tasks": lambda: handle_load_tasks(cell, tx),
            "load_open_tasks": lambda: handle_load_open_tasks(cell, tx),
            "load_task_templates": lambda: handle_load_task_templates(cell, tx),
            "use_task_template": lambda: handle_use_task_template(cell, tx),
            "add_knowledge": lambda: handle_add_knowledge(cell, tx),
            "upload_knowledge": lambda: handle_upload_knowledge(cell, tx),
            "load_knowledge": lambda: handle_load_knowledge(cell, tx),
            "get_ui": lambda: handle_get_ui(cell, tx, access_level),
        }
        if access_level == "admin":
            handlers["create_task_template"] = lambda: handle_create_task_template(cell, tx)
            handlers["delete_task_template"] = lambda: handle_delete_task_template(cell, tx)
            handlers["delete_knowledge"] = lambda: handle_delete_knowledge(cell, tx)

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
