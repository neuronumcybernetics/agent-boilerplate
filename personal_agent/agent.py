
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from neuronum import Cell
from model import get_model
from jinja2 import Environment, FileSystemLoader


# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(filename="agent.log", level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

template_env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
    
# ── Database ─────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.db")

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

# ── Auth ─────────────────────────────────────────────────────────────────────

def is_authorized(sender: str, server_host: str) -> bool:
    if sender == server_host:
        return True
    return False

# ── Handlers ─────────────────────────────────────────────────────────────────

async def route_to_agent(cell, query: str):
    agents = await cell.list_agents() or []

    candidates = []
    for a in agents:
        try:
            config = json.loads(a.get("config", "{}"))
        except Exception:
            continue
        meta = config.get("agent_meta", {})
        agent_id = meta.get("agent_id", a.get("agent_id", ""))
        creator = a.get("creator", "")
        author = a.get("author", "")
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
            })

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
    result = llm.create_chat_completion(
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
                })
        return suggestions if suggestions else None
    except Exception:
        pass
    return None


async def handle_get_answer(cell, tx: dict):
    data = tx.get("data", {})
    query = data.get("query", "")
    context = data.get("context", "")

    llm = get_model()
    messages = [{"role": "user", "content": query}]
    if context:
        messages.insert(0, {"role": "system", "content": context})
    answer = llm.create_chat_completion(messages=messages)["choices"][0]["message"]["content"]

    route = await route_to_agent(cell, query)
    suggestions = [
        {**s, "query": query, "context": context}
        for s in route
    ] if route else []

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "suggestions": suggestions}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_confirm_task(cell, tx: dict):
    data = tx.get("data", {})
    agent_id = data.get("agent_id", "")
    creator = data.get("creator", "")
    handle = data.get("handle", "")
    query = data.get("query", "")
    context = data.get("context", "")

    payload = {"handle": handle, "agent_id": agent_id, "query": query}
    if context:
        payload["context"] = context

    try:
        response = await cell.activate_tx(payload, creator)
        answer = (
            response.get("data", {}).get("json", {}).get("answer")
            or response.get("json", {}).get("answer")
            or response.get("answer")
            or json.dumps(response)
        )
    except Exception as e:
        answer = f"Failed to reach agent: {e}"

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_read_file(cell, tx: dict):
    data = tx.get("data", {})
    file_path = data.get("file_path", "")

    if not file_path or not os.path.isfile(file_path):
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"context": ""}},
            client_public_key_str=data.get("public_key", "")
        )
        return

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        text = f"Could not read file: {e}"

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"context": text[:20000]}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_save_task(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")
    query = data.get("query", "")
    answer = data.get("answer", "")
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _db()
    conn.execute(
        "INSERT INTO tasks (cell_id, query, answer, created_at) VALUES (?, ?, ?, ?)",
        (cell_id, query, answer, created_at)
    )
    conn.commit()
    conn.close()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_load_tasks(cell, tx: dict):
    data = tx.get("data", {})
    cell_id = tx.get("sender", "")

    conn = _db()
    rows = conn.execute(
        "SELECT query, answer, created_at FROM tasks WHERE cell_id = ? ORDER BY created_at DESC LIMIT 100",
        (cell_id,)
    ).fetchall()
    conn.close()

    tasks = [dict(r) for r in rows]

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"tasks": tasks}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_ui(cell, tx: dict):
    data = tx.get("data", {})
    template = template_env.get_template("agent.html")
    host = cell.host or cell.env.get("HOST", "")
    html = template.render(host=host)
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"html": html},
        client_public_key_str=data.get("public_key", "")
    )


# ── Agent ─────────────────────────────────────────────────────────────────────

async def start_agent(cell):
    async for tx in cell.sync():
        try:
            data = tx.get("data", {})
            handle = data.get("handle", None)
            sender = tx.get("sender", "")
            server_host = cell.host or cell.env.get("HOST", "")

            print(tx)

            if not is_authorized(sender, server_host):
                logging.warning(f"Access denied: '{sender}' is not authorized")
                continue

            handlers = {
                "get_answer": lambda: handle_get_answer(cell, tx),
                "confirm_task": lambda: handle_confirm_task(cell, tx),
                "read_file": lambda: handle_read_file(cell, tx),
                "save_task": lambda: handle_save_task(cell, tx),
                "load_tasks": lambda: handle_load_tasks(cell, tx),
                "get_ui": lambda: handle_get_ui(cell, tx),
            }

            handler = handlers.get(handle)
            if handler:
                await handler()

        except Exception as e:
            logging.error(f"Error: {e}")


async def main():
    async with Cell() as cell:
        await start_agent(cell)


if __name__ == "__main__":
    asyncio.run(main())
