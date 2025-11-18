# ============================================
# ashen-mcp-server — Full Stable MCP Server
# ============================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import subprocess
import os

# --------------------------------------------
# CONFIG
# --------------------------------------------
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"
BASE_URL = "https://ashen-mcp-server.onrender.com"

app = FastAPI()

# --------------------------------------------
# CORS
# --------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 0) ROOT
# ============================================
@app.get("/", status_code=200)
async def root():
    return {"status": "ok", "message": "MCP server running"}

@app.head("/", status_code=200)
async def root_head():
    return ""


# ============================================
# 1) MCP METADATA (.well-known/mcp.json)
# ============================================
@app.get("/.well-known/mcp.json")
async def mcp_metadata():

    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "analyze_text": {
                    "name": "analyze_text",
                    "description": "Analyze/summarize text using Ollama.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["summary", "analysis", "keywords"],
                                "default": "summary"
                            },
                        },
                        "required": ["text"],
                    },
                },

                "call_ollama": {
                    "name": "call_ollama",
                    "description": "Call any Ollama model with a prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "prompt": {"type": "string"}
                        },
                        "required": ["model", "prompt"],
                    },
                },

                "summarize_file": {
                    "name": "summarize_file",
                    "description": "Read file & summarize via Ollama.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        },
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "transport": "sse",
        "sse": {"url": "/sse"},
        "streamableHttp": {"url": "/mcp"},
    }

    resp = JSONResponse(metadata)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


# ============================================
# OLLAMA HELPER
# ============================================
def call_ollama_raw(model: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode('utf-8', errors='ignore')}"
        return result.stdout.decode("utf-8", errors="ignore")

    except subprocess.TimeoutExpired:
        return "[ERROR] Ollama timeout"
    except Exception as e:
        return f"[ERROR] Ollama execution failed: {str(e)}"


# ============================================
# 2) JSON-RPC HANDLER
# ============================================
async def handle_rpc(body: dict):
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # ----------------------------------------
    # (0) initialize  — REQUIRED
    # ----------------------------------------
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        }

    # ----------------------------------------
    # (1) tools/list  — REQUIRED
    # ----------------------------------------
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {
                        "name": "analyze_text",
                        "description": "Analyze/summarize text using Ollama.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "mode": {
                                    "type": "string",
                                    "enum": ["summary", "analysis", "keywords"],
                                    "default": "summary",
                                },
                            },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "call_ollama",
                        "description": "Call any Ollama model with a prompt.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "prompt": {"type": "string"},
                            },
                            "required": ["model", "prompt"],
                        },
                    },
                    {
                        "name": "summarize_file",
                        "description": "Read file & summarize via Ollama.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ]
            }
        }

    # ----------------------------------------
    # (2) tools/call — REQUIRED
    # ----------------------------------------
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # A) analyze_text
        if tool_name == "analyze_text":
            text = arguments.get("text", "")
            mode = arguments.get("mode", "summary")

            if not text.strip():
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32602, "message": "text required"}}

            if mode == "summary":
                prompt = f"Summarize:\n{text}"
            elif mode == "analysis":
                prompt = f"Analyze:\n{text}"
            elif mode == "keywords":
                prompt = f"Keywords:\n{text}"
            else:
                prompt = text

            result = call_ollama_raw("qwen2.5:7b-instruct", prompt)
            if result.startswith("[ERROR]"):
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32603, "message": result}}

            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        # B) call_ollama
        if tool_name == "call_ollama":
            model = arguments.get("model")
            prompt = arguments.get("prompt")

            if not model or not prompt:
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32602, "message": "model & prompt required"}}

            result = call_ollama_raw(model, prompt)
            if result.startswith("[ERROR]"):
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32603, "message": result}}

            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        # C) summarize_file
        if tool_name == "summarize_file":
            path = arguments.get("path")

            if not path or not os.path.exists(path):
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32602, "message": "file not found"}}

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            prompt = f"Summarize:\n{content}"
            result = call_ollama_raw("qwen2.5:7b-instruct", prompt)

            if result.startswith("[ERROR]"):
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32603, "message": result}}

            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"content": [{"type": "text", "text": result}]}
            }

        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

    # ----------------------------------------
    # Unknown method
    # ----------------------------------------
    return {"jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


# ============================================
# 3) STREAMABLE HTTP
# ============================================
@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        })

    result = await handle_rpc(body)
    response = JSONResponse(result)
    response.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return response


# ============================================
# 4) SSE (correct format)
# ============================================
async def sse_stream():
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(
        sse_stream(),
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
        }
    )
