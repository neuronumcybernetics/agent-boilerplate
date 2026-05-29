import os
import sqlite3
from datetime import datetime, timezone
from fastmcp import FastMCP

mcp = FastMCP("CRM")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crm.db")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL DEFAULT '',
            company TEXT NOT NULL DEFAULT '',
            phone TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'lead',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            summary TEXT NOT NULL,
            next_action TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (contact_id) REFERENCES contacts(id)
        )
    """)
    conn.commit()
    conn.close()

_init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_contact(conn, query: str):
    q = f"%{query.lower()}%"
    return conn.execute(
        "SELECT * FROM contacts WHERE lower(name) LIKE ? OR lower(email) LIKE ? OR lower(company) LIKE ? ORDER BY updated_at DESC LIMIT 1",
        (q, q, q)
    ).fetchone()


def _contact_dict(row) -> dict:
    if not row:
        return {}
    keys = ["id", "name", "email", "company", "phone", "status", "notes", "created_at", "updated_at"]
    return dict(zip(keys, row))


def _interaction_dict(row) -> dict:
    if not row:
        return {}
    keys = ["id", "contact_id", "type", "summary", "next_action", "created_at"]
    return dict(zip(keys, row))


@mcp.tool()
def lookup_contact(query: str) -> dict:
    """Find a contact by name, email, or company. Returns profile and last interaction."""
    with sqlite3.connect(DB_PATH) as conn:
        row = _find_contact(conn, query)
        if not row:
            return {"success": False, "error": f"No contact found matching '{query}'"}
        contact = _contact_dict(row)
        last = conn.execute(
            "SELECT * FROM interactions WHERE contact_id = ? ORDER BY created_at DESC LIMIT 1",
            (contact["id"],)
        ).fetchone()
        interaction_count = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE contact_id = ?", (contact["id"],)
        ).fetchone()[0]
    return {
        "success": True,
        "contact": contact,
        "last_interaction": _interaction_dict(last) if last else None,
        "interaction_count": interaction_count,
    }


@mcp.tool()
def create_contact(name: str, email: str = "", company: str = "", phone: str = "", status: str = "lead", notes: str = "") -> dict:
    """Add a new contact to the CRM."""
    valid_statuses = {"lead", "prospect", "customer", "churned"}
    if status not in valid_statuses:
        status = "lead"
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        # Check for duplicate email
        if email:
            existing = conn.execute("SELECT id, name FROM contacts WHERE lower(email) = ?", (email.lower(),)).fetchone()
            if existing:
                return {"success": False, "error": f"Contact with email '{email}' already exists (id={existing[0]}, name={existing[1]})"}
        cursor = conn.execute(
            "INSERT INTO contacts (name, email, company, phone, status, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, email, company, phone, status, notes, now, now)
        )
        contact_id = cursor.lastrowid
    return {"success": True, "contact_id": contact_id, "name": name, "status": status}


@mcp.tool()
def log_interaction(contact_query: str, type: str, summary: str, next_action: str = "") -> dict:
    """Log a call, email, meeting, or note against a contact."""
    valid_types = {"call", "email", "meeting", "note"}
    if type not in valid_types:
        type = "note"
    now = _now()
    with sqlite3.connect(DB_PATH) as conn:
        row = _find_contact(conn, contact_query)
        if not row:
            return {"success": False, "error": f"No contact found matching '{contact_query}'"}
        contact = _contact_dict(row)
        conn.execute(
            "INSERT INTO interactions (contact_id, type, summary, next_action, created_at) VALUES (?, ?, ?, ?, ?)",
            (contact["id"], type, summary, next_action, now)
        )
        conn.execute("UPDATE contacts SET updated_at = ? WHERE id = ?", (now, contact["id"]))
    return {
        "success": True,
        "contact": contact["name"],
        "type": type,
        "summary": summary,
        "next_action": next_action or None,
    }


@mcp.tool()
def get_pipeline(status: str = "all", limit: int = 20) -> dict:
    """Return contacts with their status, last interaction, and next action."""
    valid_statuses = {"lead", "prospect", "customer", "churned", "all"}
    if status not in valid_statuses:
        status = "all"
    limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as conn:
        if status == "all":
            rows = conn.execute(
                "SELECT * FROM contacts ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE status = ? ORDER BY updated_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        contacts = []
        for row in rows:
            c = _contact_dict(row)
            last = conn.execute(
                "SELECT * FROM interactions WHERE contact_id = ? ORDER BY created_at DESC LIMIT 1",
                (c["id"],)
            ).fetchone()
            c["last_interaction"] = _interaction_dict(last) if last else None
            contacts.append(c)
    return {"success": True, "count": len(contacts), "contacts": contacts}
