# ============================================
# ashen-mcp-server — A안 최종 안정판
# ============================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import subprocess
import os
import base64
import textwrap

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
                "local_mcp_builder": {
                    "name": "local_mcp_builder",
                    "description": "Generate a full local MCP server package (Python + PowerShell installer).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "include_ollama_setup": {"type": "boolean"},
                            "server_port": {"type": "number"},
                        },
                        "required": [],
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
    """Call Ollama model via subprocess"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minutes
        )

        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode('utf-8', errors='ignore')}"

        return result.stdout.decode("utf-8", errors="ignore")

    except subprocess.TimeoutExpired:
        return "[ERROR] Ollama timeout after 5 minutes"
    except Exception as e:
        return f"[ERROR] Ollama execution failed: {str(e)}"


# ============================================
# 2) JSON-RPC Handler
# ============================================
async def handle_rpc(body: dict):
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # ----------------------------------------
    # analyze_text
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
            prompt = f"Summarize:\n{text}"
        elif mode == "analysis":
            prompt = f"Analyze deeply:\n{text}"
        elif mode == "keywords":
            prompt = f"Extract keywords:\n{text}"
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
    # call_ollama
    # ----------------------------------------
    if method == "call_ollama":
        model = params.get("model")
        prompt = params.get("prompt")

        if not model or not prompt:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": "model and prompt required"}
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
    # summarize_file
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

        prompt = f"Summarize:\n{content}"
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
    # local_mcp_builder
    # ----------------------------------------
    if method == "local_mcp_builder":

        include_ollama = params.get("include_ollama_setup", True)
        port = params.get("server_port", 8765)

        # 1) Python server generator
        local_server_py = textwrap.dedent(f"""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        import json, asyncio

        app = FastAPI()

        @app.get("/")
        async def root():
            return {{"status": "local_mcp_running"}}

        @app.get("/.well-known/mcp.json")
        async def meta():
            return {{
                "protocolVersion": "2025-06-18",
                "capabilities": {{
                    "tools": {{
                        "echo": {{
                            "name": "echo",
                            "description": "Echo text",
                            "inputSchema": {{
                                "type": "object",
                                "properties": {{
                                    "text": {{"type": "string"}}
                                }},
                                "required": ["text"]
                            }}
                        }}
                    }}
                }},
                "serverInfo": {{"name": "local-mcp", "version": "1.0"}},
                "transport": "sse",
                "sse": {{"url": "/sse"}},
                "streamableHttp": {{"url": "/mcp"}}
            }}

        @app.post("/mcp")
        async def rpc(request: Request):
            body = await request.json()
            method = body.get("method")
            rpc_id = body.get("id")

            if method == "echo":
                text = body.get("params", {{}}).get("text", "")
                return {{
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {{
                        "content": [{{
                            "type": "text",
                            "text": text
                        }}]
                    }}
                }}

            return {{
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {{
                    "code": -32601,
                    "message": "unknown method"
                }}
            }}

        async def sse_stream():
            yield "data: {{\\"event\\": \\"connected\\"}}\\n\\n"
            while True:
                yield "data: {{\\"event\\": \\"alive\\"}}\\n\\n"
                await asyncio.sleep(5)

        @app.get("/sse")
        async def sse_ep():
            return StreamingResponse(sse_stream(), media_type="text/event-stream")
        """)

        # 2) PowerShell installer/runner
        ps1_script = textwrap.dedent(f"""
        Write-Host "Installing dependencies..." -ForegroundColor Green
        python -m pip install --quiet fastapi uvicorn

        Write-Host "Starting Local MCP Server on port {port}..." -ForegroundColor Cyan
        uvicorn local_mcp_server:app --host 0.0.0.0 --port {port} --reload
        """)

        # Packaging
        file_package = {
            "local_mcp_server.py": local_server_py,
            "run_local_mcp.ps1": ps1_script,
        }

        encoded = base64.b64encode(json.dumps(file_package).encode()).decode()

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [
                    {"type": "text", "text": "✅ Local MCP package created successfully!"},
                    {"type": "file", "name": "local_mcp_package.json", "data": encoded}
                ]
            }
        }

    # ----------------------------------------
    # Unknown RPC
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
# 4) SSE (final stable version)
# ============================================
async def sse_stream():
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
    }
    return StreamingResponse(sse_stream(), headers=headers)
