"""
LangGraph Banking Agent
- ReAct graph with tool node, MCP tools loaded via langchain-mcp-adapters
- Connects to shared MCP server with X-Agent-Key (LANGGRAPH_AGENT_KEY)
- Supabase JWT auth for frontend requests
- Checkpointed in-memory state (MemorySaver)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import dns_patch  # noqa: F401

import os
import asyncio
from typing import Annotated, TypedDict

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")
load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
AGENT_KEY = os.environ.get("LANGGRAPH_AGENT_KEY", "9631ba8000f381ff77440bd153dcd0a1")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://huhdpxvsdxnsfsqxwpux.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

app = FastAPI(title="LangGraph Banking Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

SYSTEM_PROMPT = (
    "You are a professional banking assistant powered by LangGraph. "
    "Help users manage their accounts: check balances, transfer funds, "
    "view transaction history, pay bills, open/close accounts, check loan "
    "eligibility, and convert currencies. Be concise and confirm before transfers."
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip("\"'")
GROQ_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "openai/gpt-oss-120b").strip("\"'")
if "llama-3.3" in GROQ_MODEL:
    GROQ_MODEL = "openai/gpt-oss-120b"

# ── LLMs & Fallback ──────────────────────────────────────────────────────────
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

groq_llm = None
if GROQ_API_KEY:
    try:
        from langchain_groq import ChatGroq
        groq_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
        )
    except Exception:
        pass

# Automatically fall back to Groq if Gemini 503s or fails
llm_with_fallback = gemini_llm.with_fallbacks([groq_llm]) if groq_llm else gemini_llm

def get_llm(model_preference: str = "auto"):
    if model_preference == "groq" and groq_llm:
        return groq_llm
    if model_preference == "gemini":
        return gemini_llm
    return llm_with_fallback


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Supabase JWT ──────────────────────────────────────────────────────────────
async def verify_supabase_jwt(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1]
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY}
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Supabase token")
    return resp.json()


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph(tools: list, checkpointer: MemorySaver, active_llm=None):
    model = active_llm or llm_with_fallback
    llm_with_tools = model.bind_tools(tools)

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    async def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = "auto"  # "auto" (Gemini with Groq fallback), "gemini", or "groq"


@app.get("/health")
def health():
    return {"status": "ok", "agent": "langgraph"}


# ── Shared state: one checkpointer; MCP tools and compiled graphs are reused ───
checkpointer = MemorySaver()
_tools: list | None = None
_graphs: dict[str, object] = {}
_tools_lock = asyncio.Lock()


async def get_tools(refresh: bool = False) -> list:
    """Tool discovery opens an SSE connection, so do it once; tool calls open their own sessions."""
    global _tools
    async with _tools_lock:
        if _tools is None or refresh:
            client = MultiServerMCPClient(
                {
                    "banking": {
                        "url": f"{MCP_SERVER_URL}/mcp/sse",
                        "transport": "sse",
                        "headers": {"X-Agent-Key": AGENT_KEY},
                    }
                }
            )
            _tools = await client.get_tools()
            _graphs.clear()
        return _tools


def get_graph(key: str, llm, tools: list):
    if key not in _graphs:
        _graphs[key] = build_graph(tools, checkpointer, active_llm=llm)
    return _graphs[key]


def _is_capacity_error(err_str: str) -> bool:
    return "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str


async def _run(graph, message: str, config: dict) -> str:
    result = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)
    last_msg = result["messages"][-1]
    return last_msg.content if hasattr(last_msg, "content") else str(last_msg)


@app.post("/chat")
async def chat(body: ChatRequest, user: dict = Depends(verify_supabase_jwt)):
    user_id = user.get("id", "anon")
    config = {"configurable": {"thread_id": f"{user_id}_{body.session_id}"}}

    try:
        tools = await get_tools()
    except Exception as e:
        print(f"[langgraph] MCP server unavailable: {e}")
        return {"reply": "The banking tool server is currently unavailable. Please try again shortly."}

    primary_key = body.model if body.model in ("gemini", "groq") else "auto"
    primary = get_graph(primary_key, get_llm(body.model), tools)

    reply = None
    err_str = ""
    for attempt in range(2):
        try:
            reply = await _run(primary, body.message, config)
            break
        except Exception as e:
            err_str = str(e)
            if attempt == 0 and _is_capacity_error(err_str):
                await asyncio.sleep(1.5)
                continue
            break

    if reply is None and groq_llm and body.model != "gemini" and primary_key != "groq":
        try:
            print(f"[langgraph] Error with primary LLM ({err_str[:40]}), falling back to Groq...")
            reply = await _run(get_graph("groq", groq_llm, tools), body.message, config)
        except Exception as fb_err:
            print(f"[langgraph] Groq fallback error: {fb_err}")

    if reply is None:
        print(f"[langgraph] chat failed: {err_str}")
        if _is_capacity_error(err_str):
            return {"reply": "Google Gemini is momentarily experiencing high demand (503). Please resend your message in a few seconds."}
        return {"reply": "Sorry, something went wrong processing your request. Please try again."}
    return {"reply": reply}
