from fastmcp import FastMCP
from mcp_servers.crm import mcp as crm_mcp

mcp = FastMCP("Agent Tools")
mcp.mount(crm_mcp)
