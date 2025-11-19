# ===========================================================
# ashen-mcp-server (FULL FIXED — 100% WORKING)
# ===========================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, subprocess, os, base64, textwrap

# --------------------------------------------
# CONFIG
# --------------------------------------------
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.1"
PROTOCOL_VERSION = "2025-06-18"

app = FastAPI()

# CORS (더 명시적으로)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ===========================================================
# ROOT
# ===========================================================
@app.get("/")
async def root():
    return {"status": "ok", "server": SERVER_NAME, "version": SERVER_VERSION}

@app.head("/")
async def head_root():
    return ""


# ===========================================================
# MCP METADATA
# ===========================================================
@app.get("/.well-known/mcp.json")
async def mcp_wellknown():
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "analyze_text": {},
                "call_ollama": {},
                "summarize_file": {},
                "local_mcp_builder": {},
            }
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "transport": "sse",
        "sse": {"url": "/sse"},
        "streamableHttp": {"url": "/mcp"}
    }
    return JSONResponse(metadata)


# ===========================================================
# HELPER — OLLAMA
# ===========================================================
def call_ollama_raw(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode()}"
        return result.stdout.decode()
    except Exception as e:
        return f"[ERROR] Ollama execution failed: {str(e)}"


# ===========================================================
# JSON-RPC HANDLER
# ===========================================================
async def handle_rpc(body):
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # -------------------------------------------------------
    # REQUIRED: initialize
    # -------------------------------------------------------
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                },
                "capabilities": {
                    "tools": {
                        "analyze_text": {},
                        "call_ollama": {},
                        "summarize_file": {},
                        "local_mcp_builder": {},
                    }
                }
            }
        }

    # -------------------------------------------------------
    # REQUIRED: tools/list
    # -------------------------------------------------------
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {"name": "analyze_text"},
                    {"name": "call_ollama"},
                    {"name": "summarize_file"},
                    {"name": "local_mcp_builder"}
                ]
            }
        }

    # -------------------------------------------------------
    # analyze_text
    # -------------------------------------------------------
    if method == "analyze_text":
        text = params.get("text", "")
        mode = params.get("mode", "summary")

        if not text.strip():
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32602,"message":"text required"}}

        prompts = {
            "summary": f"Summarize:\n\n{text}",
            "analysis": f"Analyze deeply:\n\n{text}",
            "keywords": f"Extract keywords:\n\n{text}"
        }
        prompt = prompts.get(mode, text)

        result = call_ollama_raw("qwen2.5:7b-instruct", prompt)

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # call_ollama
    # -------------------------------------------------------
    if method == "call_ollama":
        model = params.get("model")
        prompt = params.get("prompt")

        if not (model and prompt):
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32602,"message":"model and prompt required"}}

        result = call_ollama_raw(model, prompt)

        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "content":[{"type":"text","text":result}],
                "isError":result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # summarize_file
    # -------------------------------------------------------
    if method == "summarize_file":
        path = params.get("path")

        if not (path and os.path.exists(path)):
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32602,"message":"File not found"}}

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32603,"message":"File read error"}}

        result = call_ollama_raw("qwen2.5:7b-instruct",
                                 f"Summarize this:\n\n{content}")

        return {
            "jsonrpc":"2.0","id":rpc_id,
            "result":{
                "content":[{"type":"text","text":result}],
                "isError":result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # local_mcp_builder
    # -------------------------------------------------------
    if method == "local_mcp_builder":
        port = params.get("server_port", 8765)

        local_server_py = textwrap.dedent(f"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

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
        "serverInfo": {{"name":"local-mcp","version":"1.0"}},
        "transport":"sse",
        "sse":{{"url":"/sse"}},
        "streamableHttp":{{"url":"/mcp"}}
    }}

@app.options("/mcp")
async def mcp_options():
    return JSONResponse(content={{}})

@app.post("/mcp")
async def rpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method")

    if method == "initialize":
        return {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{
                "protocolVersion":"2025-06-18",
                "serverInfo":{{"name":"local-mcp","version":"1.0"}},
                "capabilities":{{"tools":{{"echo":{{}}}}}}
            }}
        }}

    if method == "tools/list":
        return {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{"tools":[{{"name":"echo"}}]}}
        }}

    if method == "echo":
        msg = body.get("params",{{}}).get("text","")
        return {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{"content":[{{"type":"text","text":msg}}]}}
        }}

    return {{
        "jsonrpc":"2.0",
        "id":rpc_id,
        "error":{{"code":-32601,"message":"unknown method"}}
    }}

@app.options("/sse")
async def sse_options():
    return JSONResponse(content={{}})

async def sse_stream():
    yield 'data: {{"event":"connected"}}\\n\\n'
    while True:
        yield 'data: {{"event":"alive"}}\\n\\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_ep():
    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={{
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }}
    )
        """)

        ps1_script = textwrap.dedent(f"""
# UTF-8 Encoding
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Installing dependencies..." -ForegroundColor Green
python -m pip install --quiet fastapi uvicorn

Write-Host "Starting Local MCP Server on port {port}..." -ForegroundColor Cyan
uvicorn local_mcp_server:app --host 0.0.0.0 --port {port} --reload
        """)

        readme_text = textwrap.dedent("""
# Local MCP Server

## HOW TO INSTALL
1. Extract all files
2. Open PowerShell as Administrator
3. Run: .\\run_local_mcp.ps1

## TEST
http://localhost:8765
http://localhost:8765/.well-known/mcp.json
        """)

        package = {
            "local_mcp_server.py": local_server_py,
            "run_local_mcp.ps1": ps1_script,
            "README.md": readme_text
        }

        encoded = base64.b64encode(json.dumps(package).encode()).decode()

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [
                    {"type": "text", "text": "✅ Local MCP package generated successfully!"},
                    {"type": "resource", "resource": {
                        "uri": f"data:application/json;base64,{encoded}",
                        "mimeType": "application/json"
                    }}
                ]
            }
        }

    # -------------------------------------------------------
    # Unknown method
    # -------------------------------------------------------
    return {
        "jsonrpc":"2.0",
        "id":rpc_id,
        "error":{"code":-32601,"message":f"Unknown method: {method}"}
    }


# ===========================================================
# HTTP /mcp (POST + OPTIONS for CORS)
# ===========================================================
@app.options("/mcp")
async def mcp_options():
    """CORS preflight handler"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc":"2.0",
            "id":None,
            "error":{"code":-32700,"message":"Parse error"}
        })

    return JSONResponse(await handle_rpc(body))


# ===========================================================
# SSE (GET + OPTIONS for CORS)
# ===========================================================
@app.options("/sse")
async def sse_options():
    """CORS preflight for SSE"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

async def sse_stream():
    yield 'data: {"event":"connected"}\n\n'
    while True:
        yield 'data: {"event":"alive"}\n\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
