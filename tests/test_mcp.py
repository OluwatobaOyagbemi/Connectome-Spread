"""The MCP server must expose every tool and return the app's numbers."""

import asyncio
import json

import pytest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "mcp_server.py"
EXPECTED_TOOLS = {"list_regions", "list_genotypes", "simulate_spread", "compare_genotypes",
                  "sweep_beta", "run_controls", "scan_seeds", "scan_lesions"}


async def _call(name, args):
    params = StdioServerParameters(command="python3", args=[str(SERVER)])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool(name, args)
            return tools, res


def test_tools_and_reference_result():
    tools, res = asyncio.run(_call("simulate_spread", {"seed": "entorhinal"}))
    assert tools == EXPECTED_TOOLS
    d = json.loads(res.content[0].text)
    assert d["regions_reached"] == 24
    assert d["braak_concordance_rho"] == pytest.approx(0.64, abs=0.005)
    assert d["arrivals"][0]["region"].endswith("entorhinal")


def test_unknown_region_is_a_clean_tool_error():
    _, res = asyncio.run(_call("simulate_spread", {"seed": "nowhere"}))
    assert res.is_error
    assert "unknown region" in res.content[0].text
