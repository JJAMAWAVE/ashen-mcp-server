# ============================================
# ashen-mcp-server — Full Stable Version
# ============================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import subprocess
import os

# --------------------------------------------
# SERVER CONFIG
# --------------------------------------------
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"
BASE_URL = "https://ashen-mcp-server.onrender.com"

app = FastAPI()

# --------------------------------------------
# CORS (Required for ChatGPT)
# --------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 0) ROOT (GET + HEAD)
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
                # Tool A: analyze text (Ollama)
                "analyze_text": {
                    "name": "analyze_text",
                    "description": "Analyze or summarize text using offline Ollama models.",
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
                    }
                },

                # Tool B: call_ollama (prompt 모델 호출)
                "call_ollama": {
                    "name": "call_ollama",
                    "description": "Call any local Ollama model with a prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "prompt": {"type": "string"},
                        },
                        "required": ["model", "prompt"],
                    }
                },

                # Tool C: summarize file
                "summarize_file": {
                    "name": "summarize_file",
                    "description": "Read a local file and summarize its content using Ollama.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"],
                    }
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
# Ollama Helper Function
# ============================================
def call_ollama_raw(model: str, prompt: str) -> str:
    """
    Ollama 모델을 직접 호출 (subprocess)
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )

        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode('utf-8', errors='ignore')}"

        return result.stdout.decode("utf-8", errors="ignore")

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
    # A) analyze_text
    # ----------------------------------------
    if method == "analyze_text":
        text = params.get("text", "")
        mode = params.get("mode", "summary")

        if not text.strip():
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": "text is required"}
            }

        if mode == "summary":
            prompt = f"Summarize the following text:\n\n{text}"
        elif mode == "analysis":
            prompt = f"Analyze the following text deeply:\n\n{text}"
        elif mode == "keywords":
            prompt = f"Extract important keywords from the following text:\n\n{text}"
        else:
            prompt = text

        result = call_ollama_raw("qwen2.5:7b-instruct", prompt)

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # ----------------------------------------
    # B) call_ollama
    # ----------------------------------------
    if method == "call_ollama":
        model = params.get("model")
        prompt = params.get("prompt")

        if not model or not prompt:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32602,
                    "message": "model and prompt required"
                }
            }

        result = call_ollama_raw(model, prompt)

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # ----------------------------------------
    # C) summarize_file
    # ----------------------------------------
    if method == "summarize_file":
        path = params.get("path")

        if not path or not os.path.exists(path):
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": "File not found"}
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32603, "message": "File read error"}
            }

        prompt = f"Summarize this file content:\n\n{content}"
        result = call_ollama_raw("qwen2.5:7b-instruct", prompt)

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # ----------------------------------------
    # D) Unknown Method
    # ----------------------------------------
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    }


# ============================================
# 3) STREAMABLE HTTP /mcp
# ============================================
@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        })

    result = await handle_rpc(body)
    response = JSONResponse(result)
    response.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return response


# ============================================
# 4) SSE (FIXED VERSION — instant flush)
# ============================================
async def sse_stream():
    # 첫 이벤트 즉시 전송
    yield "event: connected\ndata: {}\n\n"

    # heartbeat
    while True:
        yield "event: alive\ndata: {}\n\n"
        await asyncio.sleep(5)


@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream"
    )
