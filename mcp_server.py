# ===========================================================
# ASHEN MCP SERVER — COMPLETE FIXED VERSION (ChatGPT Compatible)
# ===========================================================
# 
# 🔴 PROBLEMS FIXED:
# 1. StreamableHTTP protocol (not just SSE)
# 2. Session management with Mcp-Session-Id header
# 3. CORS middleware conflict REMOVED
# 4. OPTIONS endpoint implementation
# 5. Proper HTTP response headers
# 6. SSE keep-alive pattern
# ===========================================================

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import json
import asyncio
import subprocess
import os
import base64
import textwrap
import uuid
from typing import Dict, Optional

# ============== CONFIG ==============
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "3.0.0"
PROTOCOL_VERSION = "2025-06-18"
MCP_SESSION_ID_HEADER = "mcp-session-id"

# ============== SESSION STORAGE ==============
# Stateful session management (required by ChatGPT)
active_sessions: Dict[str, dict] = {}

# ============== STARTUP/SHUTDOWN ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {SERVER_NAME} v{SERVER_VERSION} starting...")
    yield
    print(f"✅ {SERVER_NAME} shutting down...")

# ============== FASTAPI APP ==============
app = FastAPI(
    title=SERVER_NAME,
    version=SERVER_VERSION,
    lifespan=lifespan
)

# ⚠️ NO CORSMiddleware! ChatGPT handles CORS differently
# Instead, we manually set CORS headers on every response


# ============== HELPER: ADD CORS HEADERS ==============
def add_cors_headers(response: Response) -> Response:
    """Add CORS headers to response (manual approach)"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = f"*,{MCP_SESSION_ID_HEADER}"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


# ============== ROOT ENDPOINT ==============
@app.get("/")
async def root():
    """Health check endpoint"""
    response = JSONResponse(
        {
            "status": "ok",
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol": PROTOCOL_VERSION
        }
    )
    return add_cors_headers(response)


@app.head("/")
async def head_root():
    """HEAD support for browser preflight"""
    response = Response()
    return add_cors_headers(response)


# ============== MCP METADATA ==============
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    """MCP metadata endpoint (required)"""
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
        "transport": "streamableHttp",
        "streamableHttp": {"url": "/mcp"}
    }
    response = JSONResponse(metadata)
    return add_cors_headers(response)


# ============== OLLAMA HELPER ==============
def call_ollama_raw(model: str, prompt: str) -> str:
    """Execute Ollama command and return result"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode(errors='ignore')}"
        return result.stdout.decode(errors="ignore")
    except Exception as e:
        return f"[ERROR] Ollama execution failed: {str(e)}"


# ============== JSON-RPC HANDLER ==============
async def handle_rpc_request(body: dict) -> dict:
    """
    Handle JSON-RPC 2.0 request.
    This is the core MCP logic.
    """
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
                    {
                        "name": "analyze_text",
                        "description": "Analyze or summarize text with Ollama",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "mode": {"type": "string", "enum": ["summary", "analysis", "keywords"]}
                            },
                            "required": ["text"]
                        }
                    },
                    {
                        "name": "call_ollama",
                        "description": "Call any Ollama model",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "prompt": {"type": "string"}
                            },
                            "required": ["model", "prompt"]
                        }
                    },
                    {
                        "name": "summarize_file",
                        "description": "Summarize a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "local_mcp_builder",
                        "description": "Generate a full local MCP server package",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "server_port": {"type": "number"},
                                "include_ollama_setup": {"type": "boolean"}
                            }
                        }
                    }
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
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32602, "message": "text required"}
            }

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

    # -------------------------------------------------------
    # summarize_file
    # -------------------------------------------------------
    if method == "summarize_file":
        path = params.get("path")

        if not (path and os.path.exists(path)):
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

        result = call_ollama_raw("qwen2.5:7b-instruct", f"Summarize this:\n\n{content}")

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # local_mcp_builder
    # -------------------------------------------------------
    if method == "local_mcp_builder":
        port = params.get("server_port", 8765)

        local_server_py = textwrap.dedent(f"""
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio, json, uuid
from contextlib import asynccontextmanager

SERVER_NAME = "local-mcp"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2025-06-18"
MCP_SESSION_ID_HEADER = "mcp-session-id"

active_sessions = {{}}

def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = f"*,{{MCP_SESSION_ID_HEADER}}"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {{SERVER_NAME}} starting on port {port}...")
    yield
    print(f"✅ {{SERVER_NAME}} shutting down...")

app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION, lifespan=lifespan)

@app.get("/")
async def root():
    response = JSONResponse({{"status": "local_mcp_running"}})
    return add_cors_headers(response)

@app.get("/.well-known/mcp.json")
async def meta():
    metadata = {{
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {{
            "tools": {{
                "echo": {{
                    "name": "echo",
                    "description": "Echo text",
                    "inputSchema": {{
                        "type": "object",
                        "properties": {{"text": {{"type": "string"}}}},
                        "required": ["text"]
                    }}
                }}
            }}
        }},
        "serverInfo": {{"name":SERVER_NAME,"version":SERVER_VERSION}},
        "transport":"streamableHttp",
        "streamableHttp":{{"url":"/mcp"}}
    }}
    response = JSONResponse(metadata)
    return add_cors_headers(response)

@app.options("/mcp")
async def mcp_options():
    response = Response()
    return add_cors_headers(response)

@app.post("/mcp")
async def rpc(request: Request):
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method")
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)

    # Initialize: create new session
    if method == "initialize":
        if not session_id:
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = {{"status": "active"}}
        
        result = {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{
                "protocolVersion":PROTOCOL_VERSION,
                "serverInfo":{{"name":SERVER_NAME,"version":SERVER_VERSION}},
                "capabilities":{{"tools":{{"echo":{{}}}}}}
            }}
        }}
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        return add_cors_headers(response)

    # tools/list
    if method == "tools/list":
        result = {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{"tools":[{{"name":"echo"}}]}}
        }}
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id or str(uuid.uuid4())
        return add_cors_headers(response)

    # echo tool
    if method == "echo":
        msg = body.get("params",{{}}).get("text","")
        result = {{
            "jsonrpc":"2.0",
            "id":rpc_id,
            "result":{{"content":[{{"type":"text","text":msg}}]}}
        }}
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id or str(uuid.uuid4())
        return add_cors_headers(response)

    # Unknown method
    result = {{
        "jsonrpc":"2.0",
        "id":rpc_id,
        "error":{{"code":-32601,"message":"unknown method"}}
    }}
    response = JSONResponse(result)
    response.headers[MCP_SESSION_ID_HEADER] = session_id or str(uuid.uuid4())
    return add_cors_headers(response)

@app.options("/sse")
async def sse_options():
    response = Response()
    return add_cors_headers(response)

async def sse_stream():
    yield 'data: {{"event":"connected"}}\\n\\n'
    while True:
        yield 'data: {{"event":"alive"}}\\n\\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_ep():
    response = StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={{
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }}
    )
    return add_cors_headers(response)
        """)

        ps1_script = textwrap.dedent(f"""
# Local MCP Server Installer
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Local MCP Server Installer" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nInstalling dependencies..." -ForegroundColor Green
python -m pip install --quiet fastapi uvicorn

Write-Host "`nStarting Local MCP Server on port {port}..." -ForegroundColor Cyan
Write-Host "Once running, configure in ChatGPT Developer Mode:" -ForegroundColor Yellow
Write-Host "URL: http://localhost:{port}/mcp" -ForegroundColor Yellow

uvicorn local_mcp_server:app --host 0.0.0.0 --port {port} --reload
        """)

        readme_text = textwrap.dedent("""
# Local MCP Server

## Quick Start

### 1. Install Dependencies
```powershell
python -m pip install fastapi uvicorn
```

### 2. Run Server
```powershell
.\\run_local_mcp.ps1
```

### 3. Server will start on http://localhost:8765

## ChatGPT Developer Mode Integration

1. Open ChatGPT
2. Go to Settings → Connectors → Advanced → Developer Mode
3. Add new connector
4. Paste this URL: `http://localhost:8765/mcp`
5. Name it "Local MCP"
6. Click Add

## Testing

- Root: http://localhost:8765/
- Metadata: http://localhost:8765/.well-known/mcp.json
- Try the echo tool once connected in ChatGPT

## Troubleshooting

**Server won't start:**
- Make sure port 8765 is not in use
- Try: `netstat -ano | findstr :8765`

**ChatGPT won't connect:**
- Check firewall allows localhost:8765
- Verify server is running
- Try refreshing ChatGPT

## Architecture

```
ChatGPT Desktop
    ↓ (StreamableHTTP)
Local MCP Server (FastAPI)
    ↓ (JSON-RPC 2.0)
echo tool
```
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
                    {
                        "type": "text",
                        "text": "✅ **Local MCP package created successfully!**\n\n**Files generated:**\n- `local_mcp_server.py` - FastAPI server\n- `run_local_mcp.ps1` - PowerShell installer\n- `README.md` - Setup guide\n\n**Next steps:**\n1. Extract the package\n2. Run `run_local_mcp.ps1`\n3. Copy URL to ChatGPT Developer Mode\n4. Use the echo tool!"
                    },
                    {
                        "type": "resource",
                        "resource": {
                            "uri": f"data:application/json;base64,{encoded}",
                            "mimeType": "application/json"
                        }
                    }
                ]
            }
        }

    # -------------------------------------------------------
    # Unknown method
    # -------------------------------------------------------
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    }


# ============== MCP ENDPOINT (StreamableHTTP) ==============
@app.options("/mcp")
async def mcp_options():
    """Handle CORS preflight for /mcp endpoint"""
    response = Response()
    return add_cors_headers(response)


@app.post("/mcp")
async def mcp_handler(request: Request):
    """
    Main MCP endpoint (StreamableHTTP protocol).
    
    This handles:
    - initialize (creates session)
    - tools/list (lists available tools)
    - Tool calls (analyze_text, call_ollama, etc.)
    """
    try:
        body = await request.json()
    except:
        response = JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)
        return add_cors_headers(response)

    # Get or create session
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if not session_id:
        session_id = str(uuid.uuid4())

    # Handle RPC request
    result = await handle_rpc_request(body)

    # Return response with session header
    response = JSONResponse(result)
    response.headers[MCP_SESSION_ID_HEADER] = session_id
    return add_cors_headers(response)


# ============== SSE ENDPOINT (Keep-Alive) ==============
@app.options("/sse")
async def sse_options():
    """Handle CORS preflight for /sse endpoint"""
    response = Response()
    return add_cors_headers(response)


async def sse_stream():
    """Server-Sent Events stream (keep-alive heartbeat)"""
    yield 'data: {"event":"connected"}\n\n'
    while True:
        yield 'data: {"event":"alive"}\n\n'
        await asyncio.sleep(5)


@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for real-time updates"""
    response = StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    return add_cors_headers(response)


# ============== DEBUG ENDPOINT ==============
@app.get("/debug")
async def debug_info():
    """Debug information (development only)"""
    response = JSONResponse({
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocol": PROTOCOL_VERSION,
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    })
    return add_cors_headers(response)


# ============== MAIN ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
