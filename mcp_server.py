from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import json, asyncio, subprocess, os, base64, textwrap, uuid

SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "3.1.0"
PROTOCOL_VERSION = "2025-03-26"
MCP_SESSION_ID_HEADER = "mcp-session-id"
active_sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {SERVER_NAME} v{SERVER_VERSION} starting...")
    yield
    print(f"✅ {SERVER_NAME} shutting down...")

app = FastAPI(
    title=SERVER_NAME,
    version=SERVER_VERSION,
    lifespan=lifespan
)

def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Mcp-Session-Id, Last-Event-ID"
    response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id, Content-Type"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response

def validate_mcp_request(request: Request, require_session: bool = False):
    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(status_code=415, detail="Content-Type must be application/json")
        accept = request.headers.get("accept", "")
        if "application/json" not in accept and "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include application/json or text/event-stream")
    elif request.method == "GET":
        accept = request.headers.get("accept", "")
        if "text/event-stream" not in accept:
            raise HTTPException(status_code=406, detail="Accept must include text/event-stream for SSE")
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if require_session and not session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")
    if session_id and session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    return session_id

@app.get("/")
async def root():
    response = JSONResponse({
        "status": "ok",
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocol": PROTOCOL_VERSION,
        "mcp_endpoint": "/mcp",
        "metadata": "/.well-known/mcp.json"
    })
    return add_cors_headers(response)

@app.head("/")
async def head_root():
    response = Response()
    return add_cors_headers(response)

@app.get("/.well-known/mcp.json")
async def mcp_metadata():
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
    return add_cors_headers(response)

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

async def handle_rpc_request(body: dict) -> dict:
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

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
        # ... 동일하게 이전과 같이 패키지 생성 구현, 생략 ...
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"content": [{"type": "text", "text": "Local MCP package generated."}]}}
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

@app.options("/mcp")
async def mcp_options():
    response = Response()
    return add_cors_headers(response)

@app.post("/mcp")
async def mcp_post_handler(request: Request):
    try:
        session_id = validate_mcp_request(request, require_session=False)
        body = await request.json()
        if not session_id:
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = {"created_at": asyncio.get_event_loop().time()}
        result = await handle_rpc_request(body)
        response = JSONResponse(result)
        response.headers[MCP_SESSION_ID_HEADER] = session_id
        response.headers["Content-Type"] = "application/json"
        return add_cors_headers(response)
    except HTTPException as e:
        response = JSONResponse({"error": e.detail}, status_code=e.status_code)
        return add_cors_headers(response)
    except json.JSONDecodeError:
        response = JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, status_code=400)
        return add_cors_headers(response)
    except Exception as e:
        response = JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}, status_code=500)
        return add_cors_headers(response)

@app.delete("/mcp")
async def mcp_delete_handler(request: Request):
    session_id = validate_mcp_request(request, require_session=True)
    if session_id in active_sessions:
        del active_sessions[session_id]
    response = Response(status_code=200)
    return add_cors_headers(response)

@app.options("/sse")
async def sse_options():
    response = Response()
    return add_cors_headers(response)

async def sse_stream():
    yield 'data: {"event":"connected"}\n\n'
    while True:
        yield 'data: {"event":"alive"}\n\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint(request: Request):
    session_id = validate_mcp_request(request, require_session=True)
    response = StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    response.headers[MCP_SESSION_ID_HEADER] = session_id
    return add_cors_headers(response)

@app.get("/debug")
async def debug_info():
    response = JSONResponse({
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocol": PROTOCOL_VERSION,
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    })
    return add_cors_headers(response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        log_level="info"
    )
