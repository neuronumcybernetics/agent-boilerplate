
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import logging
from neuronum import Cell
from model import get_model
from fastmcp import Client
from mcp_servers import mcp as mcp_server

mcp_client = Client(mcp_server)


# ── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(filename="agent.log", level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

with open("agent.config", "r") as f:
    app_config = json.load(f)

AGENT_NAME = app_config.get("agent_meta", {}).get("name", "Agent")

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
    if audience and audience != "private":
        allowed = {a.strip() for a in audience.split(",") if a.strip()}
        sender_domain = sender.split("@", 1)[1] if "@" in sender else sender
        if sender in allowed or sender_domain in allowed:
            return True
    return False

# ── Agentic loop ─────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10

async def run_agent(query: str, context: str, handle: str = "") -> str:
    llm = get_model()

    async with mcp_client:
        tool_list = await mcp_client.list_tools()
        def fmt_tool(t):
            props = t.inputSchema.get("properties", {})
            required = set(t.inputSchema.get("required", []))
            args = ", ".join(
                f'{k}{"" if k in required else "?"}: {v.get("description", v.get("type", ""))}'
                for k, v in props.items()
            )
            return f'- {t.name}({args}): {t.description}'
        tools_text = "\n".join(fmt_tool(t) for t in tool_list)

        system = (
            f"You are {AGENT_NAME}, a CRM assistant. You manage contacts, log interactions, and track deals.\n"
            "Always use tools to read or write CRM data — never invent contact details or interaction history.\n"
            "To call a tool, output ONLY a raw JSON object on a single line — no explanation before or after it: "
            '{"tool": "tool_name", "args": {"arg": value}}\n'
            "When you have enough information to answer, reply in plain natural language with no JSON.\n"
            "NEVER mix text and a JSON tool call in the same reply.\n"
            "Be concise — confirm actions taken and surface the most relevant CRM data.\n"
            + (f"\nThe requested skill is '{handle}' — use the matching tool directly." if handle else "")
            + (f"\nContext:\n{context}" if context else "")
            + f"\n\nAvailable tools:\n{tools_text}"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        for _ in range(MAX_ITERATIONS):
            result = await asyncio.to_thread(llm.create_chat_completion, messages=messages)
            reply = result["choices"][0]["message"]["content"].strip()

            tool_call = None
            try:
                start = reply.index("{")
                end = reply.rindex("}") + 1
                parsed = json.loads(reply[start:end])
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_call = parsed
            except (ValueError, json.JSONDecodeError, TypeError):
                pass

            if tool_call:
                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})
                messages.append({"role": "assistant", "content": reply})
                try:
                    tool_result = await mcp_client.call_tool(tool_name, tool_args)
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' returned: {str(tool_result.data or '')}"})
                except Exception as e:
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' failed: {e}"})
            else:
                return reply

    return "I was unable to complete the task."


# ── Agent ─────────────────────────────────────────────────────────────────────

async def handle_tx(cell, tx: dict):
    try:
        data = tx.get("data", {})
        sender = tx.get("sender", "")
        server_host = cell.host or cell.env.get("HOST", "")
        agent_id = data.get("agent_id", "")
        auth = data.get("auth", "")
        api_key = auth.get("api_key", "")

        if api_key != "user_api_key":
            logging.warning(f"Access denied - Wrong API Key: '{sender}'")
            answer = "Access denied - Wrong API Key"
            await cell.tx_response(
                tx_id=tx.get("tx_id"),
                data={"json": {"answer": answer}},
                client_public_key_str=data.get("public_key", "")
            )
            return

        if not is_authorized(sender, server_host, agent_id):
            logging.warning(f"Access denied: '{sender}'")
            return

        answer = await run_agent(data.get("query", ""), data.get("context", ""), data.get("handle", ""))
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"answer": answer}},
            client_public_key_str=data.get("public_key", "")
        )

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
