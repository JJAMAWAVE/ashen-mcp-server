import asyncio
import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# =============================
# Server Constants
# =============================
BASE_URL = "https://ashen-mcp-server.onrender.com"
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"

app = FastAPI()

# =============================
# CORS (최대 호환성)
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],     # 매우 중요
    expose_headers=["*"],    # 매우 중요
)


# =============================
# Root 확인
# =============================
@app.get("/")
async def root():
    return {"status": "ok", "server": SERVER_NAME}


# =============================
# MCP 메타데이터 (no-cache + JSONResponse)
# =============================
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    payload = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "transport": "sse",  # SSE 기본
        "sse": {
            "url": f"{BASE_URL}/sse"
        },
        # Streamable HTTP도 지원
        "streamableHttp": {
            "url": f"{BASE_URL}/mcp"
        }
    }

    response = JSONResponse(payload)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


# =============================
# SSE Stream
# =============================
async def sse_stream(session_id: str):
    # 클라이언트 연결 확인 이벤트
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    await asyncio.sleep(0.2)

    # Keep Alive
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(5)


@app.get("/sse")
async def sse_endpoint(request: Request):
    session_id = request.query_params.get("session_id", "default")

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
    }

    return StreamingResponse(sse_stream(session_id), headers=headers)


# =============================
# JSON-RPC Router
# =============================
async def handle_rpc(body: Dict[str, Any]) -> Dict[str, Any]:
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    # === JSON-RPC: initialize ===
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
                "capabilities": {"tools": {}},
            },
        }

    # === JSON-RPC: tools/list ===
    if method == "tools/list":
        tools = [
            {
                "name": "hello",
                "description": "Simple greeting tool",
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

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools},
        }

    # === JSON-RPC: tools/call ===
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "hello":
            name = args.get("name", "friend")
            text = f"Hello, {name}!"
        elif tool_name == "add":
            a = args.get("a", 0)
            b = args.get("b", 0)
            text = f"{a} + {b} = {a + b}"
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": False,
            },
        }

    # === Unknown method ===
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}",
        },
    }


# =============================
# Streamable HTTP MCP Endpoint
# =============================
@app.post("/mcp")
async def mcp_http(request: Request):
    try:
        body = await request.json()
        response = await handle_rpc(body)
        return JSONResponse(response)

    except Exception as e:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e),
                },
            },
            status_code=400,
        )


# =============================
# SSE-based RPC Endpoint
# =============================
@app.post("/messages")
async def mcp_messages(request: Request):
    try:
        body = await request.json()
        response = await handle_rpc(body)

        # request_id 없는 경우: 반드시 응답해야 Render에서 drop 안 됨
        if body.get("id") is None:
            return JSONResponse({"jsonrpc": "2.0", "result": {"ok": True}})

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e),
                },
            },
            status_code=400,
        )
