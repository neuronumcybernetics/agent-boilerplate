
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import logging
import sqlite3
from neuronum import Cell
from model import get_model
from jinja2 import Environment, FileSystemLoader


# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(filename="agent.log", level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

template_env = Environment(loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))

with open("agent.config", "r") as f:
    app_config = json.load(f)
    
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

def is_authorized(sender: str, server_host: str, agent_id: str) -> bool:
    my_agent_id = app_config.get("agent_meta", {}).get("agent_id", "")
    if not agent_id or agent_id != my_agent_id:
        return False
    if sender == server_host:
        return True
    audience = app_config.get("agent_meta", {}).get("audience", "private")
    if audience == "public":
        return True
    if isinstance(audience, str) and audience != "private":
        allowed = [cell.strip() for cell in audience.split(",") if cell.strip()]
        if sender in allowed:
            return True
    return False

# ── Handlers ─────────────────────────────────────────────────────────────────


async def handle_get_answer(cell, tx: dict):
    data = tx.get("data", {})
    query = data.get("query", "")
    context = data.get("context", "")

    llm = get_model()
    messages = [{"role": "user", "content": query}]
    if context:
        messages.insert(0, {"role": "system", "content": context})
    answer = llm.create_chat_completion(messages=messages)["choices"][0]["message"]["content"]

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer}},
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
            agent_id = data.get("agent_id", "")

            print(tx)

            if not is_authorized(sender, server_host, agent_id):
                logging.warning(f"Access denied: '{sender}' is not authorized")
                continue

            handlers = {
                "get_answer": lambda: handle_get_answer(cell, tx),
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
