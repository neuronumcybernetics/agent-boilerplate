
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_access_rules (
            agent_id TEXT PRIMARY KEY,
            cells TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_auth_credentials (
            agent_id TEXT PRIMARY KEY,
            credentials TEXT NOT NULL,
            updated_at TEXT NOT NULL
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


def _generate_visualization(data: dict) -> list:
    """
    Analyze data structure and generate HTML visualizations as files.
    Returns list of file objects ready to be sent to frontend.
    """
    if not data or not isinstance(data, dict):
        return []

    files = []

    # Iterate through data from different tools/agents
    for tool_name, tool_data_list in data.items():
        if not tool_data_list or not isinstance(tool_data_list, list):
            continue

        for idx, tool_data in enumerate(tool_data_list):
            if not isinstance(tool_data, (dict, list)):
                continue

            # Check if data is a list of objects (table-like structure)
            if isinstance(tool_data, list) and len(tool_data) > 0 and isinstance(tool_data[0], dict):
                # Generate table visualization
                html = _generate_table_html(tool_data, f"{tool_name} Data")
                file_name = f"{tool_name}_table_{idx+1}" if len(tool_data_list) > 1 else f"{tool_name}_table"
                files.append({
                    "name": file_name,
                    "mime": "text/html",
                    "data": base64.b64encode(html.encode()).decode()
                })

            # Check if data is a single object - look for nested arrays or numeric values
            elif isinstance(tool_data, dict):
                # First, look for nested arrays of objects (like "contacts", "deals", "items", etc.)
                for nested_key, nested_value in tool_data.items():
                    if isinstance(nested_value, list) and len(nested_value) > 0 and isinstance(nested_value[0], dict):
                        # Found a nested array of objects - generate table
                        html = _generate_table_html(nested_value, f"{nested_key.replace('_', ' ').title()}")
                        file_name = f"{nested_key}_table"
                        files.append({
                            "name": file_name,
                            "mime": "text/html",
                            "data": base64.b64encode(html.encode()).decode()
                        })

                # Also check if the top-level object has numeric values for charts
                numeric_values = {k: v for k, v in tool_data.items() if isinstance(v, (int, float))}
                if len(numeric_values) >= 2:
                    # Generate bar chart
                    html = _generate_chart_html(numeric_values, f"{tool_name} Metrics", "bar")
                    file_name = f"{tool_name}_chart_{idx+1}" if len(tool_data_list) > 1 else f"{tool_name}_chart"
                    files.append({
                        "name": file_name,
                        "mime": "text/html",
                        "data": base64.b64encode(html.encode()).decode()
                    })

    return files


def _generate_table_html(data: list, title: str) -> str:
    """Generate HTML table from list of dicts."""
    if not data:
        return "<html><body><p>No data available</p></body></html>"

    # Get all unique keys from all objects
    all_keys = set()
    for item in data:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    headers = sorted(list(all_keys))

    rows_html = ""
    for item in data:
        row = "<tr>"
        for key in headers:
            value = item.get(key, "")
            row += f"<td>{value}</td>"
        row += "</tr>"
        rows_html += row

    headers_html = "".join(f"<th>{h}</th>" for h in headers)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        h2 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #333;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #4CAF50;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #444;
        }}
        tr:hover {{
            background: #333;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    <table>
        <thead>
            <tr>{headers_html}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    return html


def _convert_csv_to_html(csv_data: str, filename: str) -> str:
    """Convert CSV data to styled HTML table."""
    rows = [row.split(',') for row in csv_data.strip().split('\n')]
    if not rows:
        return "<html><body><p>Empty CSV file</p></body></html>"

    headers = rows[0]
    data_rows = rows[1:]

    headers_html = "".join(f"<th>{h.strip()}</th>" for h in headers)
    rows_html = ""
    for row in data_rows:
        row_html = "<tr>" + "".join(f"<td>{cell.strip()}</td>" for cell in row) + "</tr>"
        rows_html += row_html

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
        }}
        h2 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #333;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #4CAF50;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #444;
        }}
        tr:hover {{
            background: #333;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <h2>{filename}</h2>
    <table>
        <thead>
            <tr>{headers_html}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""
    return html


def _convert_image_to_html(image_data_b64: str, mime_type: str, filename: str) -> str:
    """Wrap image in styled HTML for iframe viewing."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        h2 {{
            color: #4CAF50;
            margin-bottom: 20px;
        }}
        img {{
            max-width: 100%;
            max-height: 80vh;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
    </style>
</head>
<body>
    <h2>{filename}</h2>
    <img src="data:{mime_type};base64,{image_data_b64}" alt="{filename}" />
</body>
</html>"""
    return html


def _convert_files_to_html(files: list) -> list:
    """Convert CSV and image files to HTML for consistent iframe viewing."""
    if not files:
        return files

    converted_files = []
    for file in files:
        mime = file.get("mime", "")
        name = file.get("name", "file")
        data = file.get("data", "")

        # Already HTML - keep as is
        if mime == "text/html":
            converted_files.append(file)
            continue

        # Convert CSV to HTML table
        if mime == "text/csv":
            try:
                csv_content = base64.b64decode(data).decode('utf-8')
                html_content = _convert_csv_to_html(csv_content, name)
                converted_files.append({
                    "name": name,
                    "mime": "text/html",
                    "data": base64.b64encode(html_content.encode()).decode()
                })
            except Exception as e:
                logging.warning(f"Failed to convert CSV to HTML: {e}")
                converted_files.append(file)  # Keep original if conversion fails
            continue

        # Convert images to HTML wrapper
        if mime.startswith("image/"):
            try:
                html_content = _convert_image_to_html(data, mime, name)
                converted_files.append({
                    "name": name,
                    "mime": "text/html",
                    "data": base64.b64encode(html_content.encode()).decode()
                })
            except Exception as e:
                logging.warning(f"Failed to convert image to HTML: {e}")
                converted_files.append(file)  # Keep original if conversion fails
            continue

        # Other file types - keep as is
        converted_files.append(file)

    return converted_files


def _generate_chart_html(data: dict, title: str, chart_type: str = "bar") -> str:
    """Generate HTML bar/line chart from key-value pairs."""
    labels = list(data.keys())
    values = list(data.values())

    # Create Chart.js data
    labels_json = json.dumps(labels)
    values_json = json.dumps(values)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
            margin: 0;
        }}
        h2 {{
            margin-top: 0;
            color: #4CAF50;
            text-align: center;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    <div class="chart-container">
        <canvas id="myChart"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('myChart').getContext('2d');
        new Chart(ctx, {{
            type: '{chart_type}',
            data: {{
                labels: {labels_json},
                datasets: [{{
                    label: '{title}',
                    data: {values_json},
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            color: '#fff'
                        }},
                        grid: {{
                            color: '#333'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            color: '#fff'
                        }},
                        grid: {{
                            color: '#333'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#fff'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


async def _get_agent_credentials(cell, agent_id: str) -> dict:
    """Get stored credentials for a specific agent (no encryption for business_agent)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT credentials FROM agent_auth_credentials WHERE agent_id = ?", (agent_id,)) as cur:
            row = await cur.fetchone()

    if not row:
        return {}

    try:
        # No encryption in business_agent - credentials stored as plain JSON
        return json.loads(row["credentials"]) if row["credentials"] else {}
    except Exception as e:
        logging.warning(f"Failed to parse credentials for agent {agent_id}: {e}")
        return {}


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
    target_agent = data.get("target_agent", "")

    # If target_agent is specified, route to that agent
    if target_agent:
        # Look up agent details
        raw_agents = await cell.list_agents() or []
        agent_info = None
        auth_params = []
        for a in raw_agents:
            try:
                config = json.loads(a.get("config", "{}"))
            except Exception:
                config = {}
            meta = config.get("agent_meta", {})
            agent_id = meta.get("agent_id", a.get("agent_id", ""))
            if agent_id == target_agent:
                agent_info = {
                    "agent_id": agent_id,
                    "creator": a.get("creator", ""),
                    "handle": ""  # Using default handle
                }
                # Extract auth parameters from config
                auth_params = config.get("auth", [])
                break

        if agent_info:
            # Load credentials for the target agent
            credentials = await _get_agent_credentials(cell, agent_info["agent_id"])

            # Call the specific agent with credentials and auth parameters
            result = await _call_agent(cell, agent_info["agent_id"], agent_info["creator"], agent_info["handle"], query, context, credentials, auth_params)
            answer = result["answer"]
            agent_data = result.get("data", {})
        else:
            answer = f"Error: Agent '{target_agent}' not found."
            agent_data = {}
    else:
        # Default behavior: use local LLM with knowledge base
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
        agent_data = {}  # No agent data for local LLM queries

    conversation = [{"role": "user", "text": query}, {"role": "agent", "text": answer}]
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (cell_id, query, answer, created_at, status, suggestions, conversation) VALUES (?, ?, ?, ?, 'open', '[]', ?)",
            (cell_id, query, answer, created_at, json.dumps(conversation))
        )
        task_id = cursor.lastrowid
        await db.commit()

    # Generate visualizations from agent data
    visualization_files = _generate_visualization(agent_data) if agent_data else []

    # Convert all files (CSV, images) to HTML for consistent iframe viewing
    all_files = _convert_files_to_html(visualization_files)

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "task_id": task_id, "data": agent_data, "files": all_files}},
        client_public_key_str=data.get("public_key", "")
    )


async def _call_agent(cell, agent_id: str, creator: str, handle: str, query: str, context: str, credentials: dict = None, auth_params: list = None) -> dict:
    try:
        # Build the data payload
        payload = {"handle": handle, "agent_id": agent_id, "query": query, "context": context}

        # Add auth credentials if provided, filtered by required auth parameters
        if credentials and auth_params:
            auth_dict = {}
            for param in auth_params:
                if param in credentials:
                    auth_dict[param] = credentials[param]
            if auth_dict:
                payload["auth"] = auth_dict

        result = await cell.activate_tx(
            data=payload,
            cell_id=creator
        )
        json_result = result.get("data", {}).get("json", result.get("json", {}))
        return {
            "answer": json_result.get("answer", str(result)),
            "data": json_result.get("data", {})
        }
    except Exception as e:
        return {"answer": f"Error contacting agent: {e}", "data": {}}


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

    # Look up auth parameters for all agents
    raw_agents = await cell.list_agents() or []
    agent_auth_map = {}
    for ra in raw_agents:
        try:
            config = json.loads(ra.get("config", "{}"))
        except Exception:
            config = {}
        meta = config.get("agent_meta", {})
        agent_id = meta.get("agent_id", ra.get("agent_id", ""))
        auth_params = config.get("auth", [])
        agent_auth_map[agent_id] = auth_params

    mode = data.get("mode", "parallel")
    accumulated_data = {}

    if mode == "sequential":
        steps = []
        step_context = context
        for a in agents:
            name = a.get("name", a["agent_id"])
            # Load credentials for this agent
            credentials = await _get_agent_credentials(cell, a["agent_id"])
            auth_params = agent_auth_map.get(a["agent_id"], [])
            result = await _call_agent(cell, a["agent_id"], a["creator"], a.get("handle", ""), query, step_context, credentials, auth_params)
            steps.append(f"**{name}:** {result['answer']}")
            step_context = (context + "\n\n" if context else "") + f"Previous step ({name}):\n{result['answer']}"
            # Accumulate data from this agent
            if result.get("data"):
                accumulated_data[name] = result["data"]
        answer = "\n\n".join(steps) if len(steps) > 1 else steps[0]
    else:
        # Load credentials for all agents first
        agent_credentials = await asyncio.gather(*[
            _get_agent_credentials(cell, a["agent_id"]) for a in agents
        ])
        # Get auth params for each agent
        auth_params_list = [agent_auth_map.get(a["agent_id"], []) for a in agents]
        # Call agents with their respective credentials and auth params
        results = await asyncio.gather(*[
            _call_agent(cell, a["agent_id"], a["creator"], a.get("handle", ""), query, context, creds, auth_params)
            for a, creds, auth_params in zip(agents, agent_credentials, auth_params_list)
        ])
        agent_names = [a.get("name", a["agent_id"]) for a in agents]
        # Accumulate data from all agents
        for name, result in zip(agent_names, results):
            if result.get("data"):
                accumulated_data[name] = result["data"]
        # Format answers
        if len(agents) == 1:
            answer = results[0]["answer"]
        else:
            answer = "\n\n".join(f"**{name}:** {r['answer']}" for name, r in zip(agent_names, results))

    conversation = [{"role": "user", "text": query}, {"role": "agent", "text": answer}]
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO tasks (cell_id, query, answer, created_at, status, suggestions, conversation) VALUES (?, ?, ?, ?, 'open', '[]', ?)",
            (cell_id, query, answer, created_at, json.dumps(conversation))
        )
        task_id = cursor.lastrowid
        await db.commit()

    # Generate visualizations from accumulated agent data
    visualization_files = _generate_visualization(accumulated_data) if accumulated_data else []

    # Convert all files (CSV, images) to HTML for consistent iframe viewing
    all_files = _convert_files_to_html(visualization_files)

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"answer": answer, "task_id": task_id, "data": accumulated_data, "files": all_files}},
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


async def handle_get_agent_access_rules(cell, tx: dict):
    data = tx.get("data", {})
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT agent_id, cells, updated_at FROM agent_access_rules") as cur:
            rows = await cur.fetchall()
    rules = {r["agent_id"]: {"cells": json.loads(r["cells"] or "[]"), "updated_at": r["updated_at"]} for r in rows}
    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"rules": rules}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_update_agent_access_rules(cell, tx: dict):
    data = tx.get("data", {})
    agent_id = data.get("target_agent_id", "")
    cells = data.get("cells", [])

    if not agent_id:
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"ok": False, "error": "agent_id is required"}},
            client_public_key_str=data.get("public_key", "")
        )
        return

    updated_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        # Use REPLACE to insert or update
        await db.execute(
            "REPLACE INTO agent_access_rules (agent_id, cells, updated_at) VALUES (?, ?, ?)",
            (agent_id, json.dumps(cells), updated_at)
        )
        await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_update_agent_auth_credentials(cell, tx: dict):
    data = tx.get("data", {})
    agent_id = data.get("target_agent_id", "")
    credentials = data.get("credentials", {})

    if not agent_id:
        await cell.tx_response(
            tx_id=tx.get("tx_id"),
            data={"json": {"ok": False, "error": "agent_id is required"}},
            client_public_key_str=data.get("public_key", "")
        )
        return

    updated_at = datetime.now(timezone.utc).isoformat()

    # Store credentials as plain JSON (no encryption in business_agent)
    credentials_json = json.dumps(credentials)

    async with aiosqlite.connect(DB_PATH) as db:
        # Use REPLACE to insert or update
        await db.execute(
            "REPLACE INTO agent_auth_credentials (agent_id, credentials, updated_at) VALUES (?, ?, ?)",
            (agent_id, credentials_json, updated_at)
        )
        await db.commit()

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"ok": True}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_agent_auth_credentials(cell, tx: dict):
    data = tx.get("data", {})

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT agent_id, credentials FROM agent_auth_credentials") as cur:
            rows = await cur.fetchall()

    # Return credentials for each agent
    credentials = {}
    for r in rows:
        agent_id = r["agent_id"]
        try:
            # No encryption in business_agent - credentials stored as plain JSON
            credentials[agent_id] = json.loads(r["credentials"]) if r["credentials"] else {}
        except Exception as e:
            logging.warning(f"Failed to parse credentials for agent {agent_id}: {e}")
            credentials[agent_id] = {}

    await cell.tx_response(
        tx_id=tx.get("tx_id"),
        data={"json": {"credentials": credentials}},
        client_public_key_str=data.get("public_key", "")
    )


async def handle_get_ui(cell, tx: dict, access_level: str = "employee"):
    data = tx.get("data", {})
    template = template_env.get_template("agent.html")
    host = cell.host or cell.env.get("HOST", "")
    operator = cell.env.get("OPERATOR", "")
    sender = tx.get("sender", "")
    all_cells = await cell.list_cells(True) or []
    cells = [c for c in all_cells if "@" not in c.get("cell_id", "")]
    employees = [c for c in all_cells if "@" in c.get("cell_id", "")]
    raw_agents = await cell.list_agents() or []

    # Load agent access rules
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT agent_id, cells FROM agent_access_rules") as cur:
            rules_rows = await cur.fetchall()
    access_rules = {r["agent_id"]: json.loads(r["cells"] or "[]") for r in rules_rows}

    # Build complete agent list (for admin management UI)
    all_agents = []
    for a in raw_agents:
        try:
            config = json.loads(a.get("config", "{}"))
        except Exception:
            config = {}
        meta = config.get("agent_meta", {})
        agent_id = meta.get("agent_id", a.get("agent_id", ""))
        legals = config.get("legals", {})
        auth = config.get("auth", {})
        all_agents.append({
            "agent_id": agent_id,
            "name": meta.get("name", a.get("agent_id", "")),
            "logo": meta.get("logo", ""),
            "description": meta.get("description", ""),
            "creator": a.get("creator", ""),
            "author": a.get("author", ""),
            "verified": bool(a.get("verified", 0)),
            "auth": auth,
            "terms": legals.get("terms", ""),
            "privacy_policy": legals.get("privacy_policy", ""),
        })

    # Build filtered agent list (based on access rules)
    # Note: Admin must also grant themselves access via agent access rules
    agents = []
    for agent in all_agents:
        agent_id = agent["agent_id"]

        # Apply access rules for all users (including admin)
        allowed_cells = access_rules.get(agent_id, [])
        # By default agents are not visible - only show if:
        # 1. "__all__" is in allowed list (all employees can access), OR
        # 2. sender is explicitly in allowed list
        if not allowed_cells or ("__all__" not in allowed_cells and sender not in allowed_cells):
            continue
        agents.append(agent)

    html = template.render(host=host, operator=operator, sender=sender, cells=cells, employees=employees, agents=agents, all_agents=all_agents, access_level=access_level)
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
        print(tx)

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
            handlers["get_agent_access_rules"] = lambda: handle_get_agent_access_rules(cell, tx)
            handlers["update_agent_access_rules"] = lambda: handle_update_agent_access_rules(cell, tx)
            handlers["update_agent_auth_credentials"] = lambda: handle_update_agent_auth_credentials(cell, tx)
            handlers["get_agent_auth_credentials"] = lambda: handle_get_agent_auth_credentials(cell, tx)

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
