# ===========================================================
# ASHEN MCP SERVER — FINAL PRODUCTION VERSION (Fixed)
# ===========================================================

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from typing import Optional
import json
import asyncio
import subprocess
import os
import base64
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
SERVER_VERSION = "3.2.2"
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
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id, Last-Event-ID"
    response.headers["Access-Control-Expose-Headers"] = f"Mcp-Session-Id, Content-Type"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


def log_request(request: Request, body: str = None):
    log_msg = f"📥 REQUEST: {request.method} {request.url.path}\n"
    log_msg += f"   Headers: {dict(request.headers)}\n"
    if body:
        log_msg += f"   Body: {body[:500]}...\n" if len(body) > 500 else f"   Body: {body}\n"
    logging.info(log_msg)


def log_response(response: Response, details: str = None):
    log_msg = f"📤 RESPONSE: status={response.status_code}\n"
    log_msg += f"   Headers: {dict(response.headers)}\n"
    if details:
        log_msg += f"   Details: {details}\n"
    logging.info(log_msg)


# ============== HELPER: VALIDATE REQUEST ==============
def validate_mcp_request(request: Request, require_session: bool = False) -> Optional[str]:
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
    response = Response()
    return add_cors_headers(response)


# ============== MCP METADATA ==============
@app.get("/.well-known/mcp.json")
async def mcp_metadata(request: Request):
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
                     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
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
