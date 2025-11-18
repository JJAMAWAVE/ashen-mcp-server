from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
BASE_URL = "https://ashen-mcp-server.onrender.com"
PROTOCOL_VERSION = "2025-06-18"

app = FastAPI()

# ============================
# CORS 설정
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 1) 루트 경로 (HEAD 포함)
# ============================
@app.get("/", status_code=200)
async def root_get():
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
            "url": "/sse"
        },
        # StreamableHTTP는 선택 사항
        "streamableHttp": {
            "url": "/mcp"
        }
    }

    resp = JSONResponse(metadata)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


# ============================
# 3) JSON-RPC 처리
# ============================
async def handle_rpc(body: dict) -> dict:
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # initialize
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

    # tools/list
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
                            "properties": {
                                "name": {"type": "string"}
                            },
                            "required": ["name"]
                        }
                    },
                    {
                        "name": "add",
                        "description": "Add two numbers",
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

    # tools/call
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "hello":
            name = args.get("name", "friend")
            text = f"Hello, {name}!"

        elif tool_name == "add":
            a = args.get("a", 0)
            b = args.get("b", 0)
            text = f"{a} + {b} = {a+b}"

        else:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}"
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

    # unknown method
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }


# ============================
# 4) Streamable HTTP 방식 (선택적)
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

    response = JSONResponse(await handle_rpc(body))
    response.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return response


# ============================
# 5) SSE Transport
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


# ============================
# 6) ★ 필수: /messages (SSE 전용 JSON-RPC 엔드포인트)
# ============================
@app.post("/messages")
async def rpc_messages(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)

    return JSONResponse(await handle_rpc(body))
