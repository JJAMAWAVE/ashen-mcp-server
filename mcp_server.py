from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
BASE_URL = "https://ashen-mcp-server.onrender.com"
PROTOCOL_VERSION = "2025-06-18"

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 1) 루트 (HEAD 포함)
# ============================
@app.get("/", status_code=200)
async def root():
    return {"status": "ok", "message": "MCP server running"}

@app.head("/", status_code=200)
async def root_head():
    return ""


# ============================
# 2) MCP Metadata
# ============================
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "transport": "sse",
        "sse": {
            "url": f"{BASE_URL}/sse"
        },
        "streamableHttp": {
            "url": f"{BASE_URL}/mcp"
        }
    }

    resp = JSONResponse(metadata)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ============================
# 3) JSON-RPC 처리기
# ============================
async def handle_rpc(body: dict) -> dict:
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # 1. initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }

    # 2. tools/list
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {
                        "name": "hello",
                        "description": "Returns a greeting",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"]
                        }
                    },
                    {
                        "name": "add",
                        "description": "Adds two numbers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                ]
            }
        }

    # 3. tools/call
    if method == "tools/call":
        tool = params.get("name")
        args = params.get("arguments", {})

        if tool == "hello":
            name = args.get("name", "friend")
            text = f"Hello, {name}!"

        elif tool == "add":
            a = args.get("a", 0)
            b = args.get("b", 0)
            text = f"{a} + {b} = {a+b}"

        else:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool}"
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": False
            }
        }

    # 알 수 없는 메서드
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }


# ============================
# 4) Streamable HTTP (선택사항)
# ============================
@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)

    result = await handle_rpc(body)
    response = JSONResponse(result)
    response.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return response


# ============================
# 5) SSE
# ============================
async def sse_stream():
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
    }
    return StreamingResponse(sse_stream(), headers=headers)
