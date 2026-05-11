from fastmcp import FastMCP
from mcp_servers.knowledge import mcp as knowledge_mcp

mcp = FastMCP("Agent Tools")
mcp.mount(knowledge_mcp)
