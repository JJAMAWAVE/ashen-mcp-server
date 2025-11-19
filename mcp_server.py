# ===========================================================
# ASHEN MCP SERVER — FINAL PRODUCTION VERSION (Fixed)
# ===========================================================
# 
# ✅ FIXES:
# 1. Render.com static file mounting conflict
# 2. Exact MCP StreamableHTTP spec compliance
# 3. Content-Type validation (406/415 errors)
# 4. Accept header validation
# 5. Proper session lifecycle
# 6. DELETE method support
# 7. Complete request/response logging
# 8. Fixed triple-quote string nesting issue
# ===========================================================

from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Optional
import json
import asyncio
import subprocess
import os
import base64
import textwrap
import uuid
import logging

# ============== LOGGING SETUP ==============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("mcp_server_requests.log"),
        logging.StreamHandler()
    ]
)

# ============== CONFIG ==============
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "3.2.1"
PROTOCOL_VERSION = "2025-03-26"
MCP_SESSION_ID_HEADER = "mcp-session-id"

# ============== SESSION STORAGE ==============
active_sessions = {}

# ============== STARTUP/SHUTDOWN ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"===== SERVER START: {SERVER_NAME} v{SERVER_VERSION} =====")
    logging.info(f"📡 Protocol: {PROTOCOL_VERSION}")
    logging.info(f"🔗 MCP Endpoint: /mcp")
    yield
    logging.info(f"===== SERVER SHUTDOWN =====")

# ============== FASTAPI APP ==============
app = FastAPI(
    title=SERVER_NAME,
    version=SERVER_VERSION,
    lifespan=lifespan
)


# ============== HELPER: ADD CORS HEADERS ==============
def add_cors_headers(response: Response) -> Response:
    """Add MCP-compliant CORS headers"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id, Last-Event-ID"
    response.headers["Access-Control-Expose-Headers"] = f"Mcp-Session-Id, Content-Type"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


def log_request(request: Request, body: str = None):
    """Log incoming request details"""
    log_msg = f"📥 REQUEST: {request.method} {request.url.path}\n"
    log_msg += f"   Headers: {dict(request.headers)}\n"
    if body:
        log_msg += f"   Body: {body[:500]}...\n" if len(body) > 500 else f"   Body: {body}\n"
    logging.info(log_msg)


def log_response(response: Response, details: str = None):
    """Log outgoing response details"""
    log_msg = f"📤 RESPONSE: status={response.status_code}\n"
    log_msg += f"   Headers: {dict(response.headers)}\n"
    if details:
        log_msg += f"   Details: {details}\n"
    logging.info(log_msg)


# ============== HELPER: VALIDATE REQUEST ==============
def validate_mcp_request(request: Request, require_session: bool = False) -> Optional[str]:
    """
    Validate MCP request headers.
    Returns session_id if valid, raises HTTPException if invalid.
    """
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    
    accept = request.headers.get("accept", "")
    if request.method == "POST":
        if "application/json" not in accept and "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include application/json or text/event-stream")
    elif request.method == "GET":
        if "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include text/event-stream for SSE")
    
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    
    if require_session and not session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")
    
    if session_id and session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    
    return session_id


# ============== ROOT ENDPOINT ==============
@app.get("/")
async def root(request: Request):
    """Health check endpoint"""
    log_request(request)
    response = JSONResponse({
        "status": "ok",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocol": PROTOCOL_VERSION,
        "mcp_endpoint": "/mcp",
        "metadata": "/.well-known/mcp.json"
    })
    response = add_cors_headers(response)
    log_response(response, "Health check")
    return response


@app.head("/")
async def head_root():
    """HEAD support for browser preflight"""
    response = Response()
    return add_cors_headers(response)


# ============== MCP METADATA ==============
@app.get("/.well-known/mcp.json")
async def mcp_metadata(request: Request):
    """MCP metadata endpoint (discovery)"""
    log_request(request)
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "analyze_text": {
                    "name": "analyze_text",
                    "description": "Analyze or summarize text with Ollama",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "mode": {"type": "string", "enum": ["summary", "analysis", "keywords"], "default": "summary"}
                        },
                        "required": ["text"]
                    }
                },
                "call_ollama": {
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
                "summarize_file": {
                    "name": "summarize_file",
                    "description": "Summarize a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                },
                "local_mcp_builder": {
                    "name": "local_mcp_builder",
                    "description": "Generate a full local MCP server package",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "server_port": {"type": "number", "default": 8765},
                            "include_ollama_setup": {"type": "boolean", "default": True}
                        }
                    }
                }
            }
        },
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "transport": "streamableHttp",
        "streamableHttp": {"url": "/mcp"}
    }
    response = JSONResponse(metadata)
    response = add_cors_headers(response)
    log_response(response, "Metadata sent")
    return response


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
    """Handle JSON-RPC 2.0 request"""
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    logging.info(f"🔧 RPC Call: method={method}, id={rpc_id}")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {}}
            }
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {"name": "analyze_text", "description": "Analyze or summarize text with Ollama",
                     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "mode": {"type": "string"}}, "required": ["text"]}},
                    {"name": "call_ollama", "description": "Call any Ollama model",
                     "inputSchema": {"type": "object", "properties": {"model": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["model", "prompt"]}},
                    {"name": "summarize_file", "description": "Summarize a file",
                     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
                    {"name": "local_mcp_builder", "description": "Generate a full local MCP server package",
                     "inputSchema": {"type": "object", "properties": {"server_port": {"type": "number"}, "include_ollama_setup": {"type": "boolean"}}}}
                ]
            }
        }

    if method == "analyze_text":
        text = params.get("text", "")
        mode = params.get("mode", "summary")
        if not text.strip():
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "text required"}}
        prompts = {"summary": f"Summarize:\n\n{text}", "analysis": f"Analyze deeply:\n\n{text}", "keywords": f"Extract keywords:\n\n{text}"}
        prompt = prompts.get(mode, text)
        result = call_ollama_raw("qwen2.5:7b-instruct", prompt)
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": result}], "isError": result.startswith("[ERROR]")}}

    if method == "call_ollama":
        model = params.get("model")
        prompt = params.get("prompt")
        if not (model and prompt):
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "model and prompt required"}}
        result = call_ollama_raw(model, prompt)
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": result}], "isError": result.startswith("[ERROR]")}}

    if method == "summarize_file":
        path = params.get("path")
        if not (path and os.path.exists(path)):
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "File not found"}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32603, "message": "File read error"}}
        result = call_ollama_raw("qwen2.5:7b-instruct", f"Summarize this:\n\n{content}")
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": result}], "isError": result.startswith("[ERROR]")}}

    if method == "local_mcp_builder":
        port = params.get("server_port", 8765)
        
        # ✅ FIX: Use single quotes and proper escaping
        local_server_code = '''from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio, json, uuid

SERVER_NAME = "local-mcp"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2025-03-26"
MCP_SESSION_ID_HEADER = "mcp-session-id"
active_sessions = {}

def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id"
    response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id, Content-Type"
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {SERVER_NAME} starting...")
    yield
    print(f"✅ {SERVER_NAME} shutting down...")

app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION, lifespan=lifespan)

@app.get("/")
async def root():
    return add_cors_headers(JSONResponse({"status": "ok", "version": SERVER_VERSION}))

@app.get("/.well-known/mcp.json")
async def meta():
    metadata = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"echo": {}}}, 
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "transport": "streamableHttp", "streamableHttp": {"url": "/mcp"}}
    return add_cors_headers(JSONResponse(metadata))

@app.options("/mcp")
async def mcp_options():
    return add_cors_headers(Response())

@app.post("/mcp")
async def rpc(request: Request):
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    body = await request.json()
    rpc_id, method = body.get("id"), body.get("method")
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if not session_id:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {"status": "active"}
    
    if method == "initialize":
        result = {"jsonrpc": "2.0", "id": rpc_id, "result": {"protocolVersion": PROTOCOL_VERSION, 
                  "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}, "capabilities": {"tools": {}}}}
    elif method == "tools/list":
        result = {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": [{"name": "echo", "description": "Echo text",
                  "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}}
    elif method == "echo":
        msg = body.get("params", {}).get("text", "")
        result = {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": f"Echo: {msg}"}]}}
    else:
        result = {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "unknown method"}}
    
    response = JSONResponse(result)
    response.headers[MCP_SESSION_ID_HEADER] = session_id
    return add_cors_headers(response)

@app.delete("/mcp")
async def delete_session(request: Request):
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    return add_cors_headers(Response(status_code=200))

async def sse_stream():
    yield 'data: {"event":"connected"}' + chr(10) + chr(10)
    while True:
        yield 'data: {"event":"alive"}' + chr(10) + chr(10)
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_ep():
    return add_cors_headers(StreamingResponse(sse_stream(), media_type="text/event-stream",
                                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}))
'''
        
        ps1_content = f'''$ErrorActionPreference = "Stop"
Write-Host "Local MCP Server Installer" -ForegroundColor Green
python -m pip install --quiet fastapi uvicorn
Write-Host "Starting server on port {port}..." -ForegroundColor Cyan
uvicorn local_mcp_server:app --host 127.0.0.1 --port {port} --reload
'''
        
        readme_content = f'''# Local MCP Server

## Quick Start
1. Install: `python -m pip install fastapi uvicorn`
2. Run: `.\\run_local_mcp.ps1`
3. Server: http://localhost:{port}

## ChatGPT Integration
Add connector URL: `http://localhost:{port}/mcp`
'''
        
        package = {
            "local_mcp_server.py": local_server_code,
            "run_local_mcp.ps1": ps1_content,
            "README.md": readme_content
        }
        
        encoded = base64.b64encode(json.dumps(package).encode()).decode()
        
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [
                    {"type": "text", "text": f"✅ Local MCP package created! Port: {port}"},
                    {"type": "resource", "resource": {"uri": f"data:application/json;base64,{encoded}", "mimeType": "application/json"}}
                ]
            }
        }

    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


# ============== MCP ENDPOINT (StreamableHTTP) ==============
@app.options("/mcp")
async def mcp_options(request: Request):
    log_request(request)
    response = Response()
    response = add_cors_headers(response)
    log_response(response, "OPTIONS preflight")
    return response


@app.post("/mcp")
async def mcp_post_handler(request: Request):
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="ignore")
        log_request(request, body_text)
        
        session_id = validate_mcp_request(request, require_session=False)
        body = json.loads(body_text)
        
        if not session_id:
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = {"created_at": asyncio.get_event_loop().time()}
            logging.info(f"✨ New session: {session_id}")
        
        result = await handle_rpc_request(body)
        
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        response.headers["Content-Type"] = "application/json"
        response = add_cors_headers(response)
        log_response(response, f"RPC: {body.get('method')}, session: {session_id[:8]}...")
        return response
        
    except HTTPException as e:
        logging.error(f"❌ HTTP Exception: {e.status_code} - {e.detail}")
        response = JSONResponse({"error": e.detail}, status_code=e.status_code)
        return add_cors_headers(response)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON Parse Error: {str(e)}")
        response = JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, status_code=400)
        return add_cors_headers(response)
    except Exception as e:
        logging.error(f"❌ Internal Error: {str(e)}", exc_info=True)
        response = JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}, status_code=500)
        return add_cors_headers(response)


@app.delete("/mcp")
async def mcp_delete_handler(request: Request):
    try:
        log_request(request)
        session_id = validate_mcp_request(request, require_session=True)
        if session_id in active_sessions:
            del active_sessions[session_id]
            logging.info(f"🗑️ Session deleted: {session_id[:8]}...")
        response = Response(status_code=200)
        return add_cors_headers(response)
    except HTTPException as e:
        return add_cors_headers(JSONResponse({"error": e.detail}, status_code=e.status_code))


@app.get("/debug")
async def debug_info(request: Request):
    log_request(request)
    response = JSONResponse({"server": SERVER_NAME, "version": SERVER_VERSION, "protocol": PROTOCOL_VERSION,
                            "active_sessions": len(active_sessions), "sessions": list(active_sessions.keys())})
    return add_cors_headers(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), log_level="info")
# ===========================================================
# ASHEN MCP SERVER — FINAL PRODUCTION VERSION
# ===========================================================
# 
# ✅ FIXES:
# 1. Render.com static file mounting conflict
# 2. Exact MCP StreamableHTTP spec compliance
# 3. Content-Type validation (406/415 errors)
# 4. Accept header validation
# 5. Proper session lifecycle
# 6. DELETE method support
# 7. Complete request/response logging
# ===========================================================

from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Optional
import json
import asyncio
import subprocess
import os
import base64
import textwrap
import uuid
import logging

# ============== LOGGING SETUP ==============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("mcp_server_requests.log"),
        logging.StreamHandler()
    ]
)

# ============== CONFIG ==============
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "3.2.0"
PROTOCOL_VERSION = "2025-03-26"
MCP_SESSION_ID_HEADER = "mcp-session-id"

# ============== SESSION STORAGE ==============
active_sessions = {}

# ============== STARTUP/SHUTDOWN ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"===== SERVER START: {SERVER_NAME} v{SERVER_VERSION} =====")
    logging.info(f"📡 Protocol: {PROTOCOL_VERSION}")
    logging.info(f"🔗 MCP Endpoint: /mcp")
    yield
    logging.info(f"===== SERVER SHUTDOWN =====")

# ============== FASTAPI APP ==============
app = FastAPI(
    title=SERVER_NAME,
    version=SERVER_VERSION,
    lifespan=lifespan
)

# ⚠️ NO CORSMiddleware! Manual header management


# ============== HELPER: ADD CORS HEADERS ==============
def add_cors_headers(response: Response) -> Response:
    """Add MCP-compliant CORS headers"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id, Last-Event-ID"
    response.headers["Access-Control-Expose-Headers"] = f"Mcp-Session-Id, Content-Type"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


def log_request(request: Request, body: str = None):
    """Log incoming request details"""
    log_msg = f"📥 REQUEST: {request.method} {request.url.path}\n"
    log_msg += f"   Headers: {dict(request.headers)}\n"
    if body:
        log_msg += f"   Body: {body[:500]}...\n" if len(body) > 500 else f"   Body: {body}\n"
    logging.info(log_msg)


def log_response(response: Response, details: str = None):
    """Log outgoing response details"""
    log_msg = f"📤 RESPONSE: status={response.status_code}\n"
    log_msg += f"   Headers: {dict(response.headers)}\n"
    if details:
        log_msg += f"   Details: {details}\n"
    logging.info(log_msg)


# ============== HELPER: VALIDATE REQUEST ==============
def validate_mcp_request(request: Request, require_session: bool = False) -> Optional[str]:
    """
    Validate MCP request headers.
    Returns session_id if valid, raises HTTPException if invalid.
    """
    # Validate Content-Type for POST
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if request.method == "POST":
        # POST must accept JSON or SSE
        if "application/json" not in accept and "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include application/json or text/event-stream")
    elif request.method == "GET":
        # GET (SSE) must accept text/event-stream
        if "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include text/event-stream for SSE")
    
    # Get session ID
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    
    # Check if session required
    if require_session and not session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")
    
    # Validate session exists (if provided)
    if session_id and session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    
    return session_id


# ============== ROOT ENDPOINT ==============
@app.get("/")
async def root(request: Request):
    """Health check endpoint"""
    log_request(request)
    response = JSONResponse({
        "status": "ok",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocol": PROTOCOL_VERSION,
        "mcp_endpoint": "/mcp",
        "metadata": "/.well-known/mcp.json"
    })
    response = add_cors_headers(response)
    log_response(response, "Health check")
    return response


@app.head("/")
async def head_root():
    """HEAD support for browser preflight"""
    response = Response()
    return add_cors_headers(response)


# ============== MCP METADATA ==============
@app.get("/.well-known/mcp.json")
async def mcp_metadata(request: Request):
    """MCP metadata endpoint (discovery)"""
    log_request(request)
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "analyze_text": {
                    "name": "analyze_text",
                    "description": "Analyze or summarize text with Ollama",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["summary", "analysis", "keywords"],
                                "default": "summary"
                            }
                        },
                        "required": ["text"]
                    }
                },
                "call_ollama": {
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
                "summarize_file": {
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
                "local_mcp_builder": {
                    "name": "local_mcp_builder",
                    "description": "Generate a full local MCP server package",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "server_port": {"type": "number", "default": 8765},
                            "include_ollama_setup": {"type": "boolean", "default": True}
                        }
                    }
                }
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
    response = add_cors_headers(response)
    log_response(response, "Metadata sent")
    return response


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
    """Handle JSON-RPC 2.0 request"""
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    logging.info(f"🔧 RPC Call: method={method}, id={rpc_id}")

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
                    "tools": {}
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
                                "mode": {"type": "string"}
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

        # Generate complete local MCP server code
        local_server_py = """# Local MCP Server - Full Implementation
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio, json, uuid

SERVER_NAME = "local-mcp"
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2025-03-26"
MCP_SESSION_ID_HEADER = "mcp-session-id"

active_sessions = {}

def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id"
    response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id, Content-Type"
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {SERVER_NAME} starting...")
    yield
    print(f"✅ {SERVER_NAME} shutting down...")

app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION, lifespan=lifespan)

@app.get("/")
async def root():
    response = JSONResponse({"status": "local_mcp_running", "version": SERVER_VERSION})
    return add_cors_headers(response)

@app.get("/.well-known/mcp.json")
async def meta():
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"echo": {}}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "transport": "streamableHttp",
        "streamableHttp": {"url": "/mcp"}
    }
    response = JSONResponse(metadata)
    return add_cors_headers(response)

@app.options("/mcp")
async def mcp_options():
    response = Response()
    return add_cors_headers(response)

@app.post("/mcp")
async def rpc(request: Request):
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    
    body = await request.json()
    rpc_id = body.get("id")
    method = body.get("method")
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)

    if not session_id:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {"status": "active"}
    
    if method == "initialize":
        result = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {}}
            }
        }
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        return add_cors_headers(response)

    if method == "tools/list":
        result = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [{
                    "name": "echo",
                    "description": "Echo text back",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]
                    }
                }]
            }
        }
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        return add_cors_headers(response)

    if method == "echo":
        msg = body.get("params", {}).get("text", "")
        result = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"content": [{"type": "text", "text": f"Echo: {msg}"}]}
        }
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        return add_cors_headers(response)

    result = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": "unknown method"}
    }
    response = JSONResponse(result)
    response.headers[MCP_SESSION_ID_HEADER] = session_id
    return add_cors_headers(response)

@app.delete("/mcp")
async def delete_session(request: Request):
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    response = Response(status_code=200)
    return add_cors_headers(response)

@app.options("/sse")
async def sse_options():
    response = Response()
    return add_cors_headers(response)

async def sse_stream():
    yield 'data: {"event":"connected"}\\n\\n'
    while True:
        yield 'data: {"event":"alive"}\\n\\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_ep():
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
"""

        ps1_script = f"""# Local MCP Server Installer
$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Local MCP Server Installer v1.0" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
python -m pip install --quiet fastapi uvicorn

Write-Host "`n✅ Dependencies installed!" -ForegroundColor Green
Write-Host "`nStarting Local MCP Server on port {port}..." -ForegroundColor Cyan
Write-Host "`n📡 MCP Endpoint: http://localhost:{port}/mcp" -ForegroundColor Yellow
Write-Host "`nPress Ctrl+C to stop the server`n" -ForegroundColor Gray

uvicorn local_mcp_server:app --host 127.0.0.1 --port {port} --reload
"""

        readme_text = f"""# Local MCP Server

## Quick Start

### 1. Install Dependencies

