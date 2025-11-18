import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# JSON-RPC over SSE용 큐
client_queue = asyncio.Queue()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP metadata
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "ashen-mcp",
        "version": "1.0.0",
        "protocolVersion": "2025-06-18",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "jsonrpc": "2.0",
    }


# ======================================================================
# SSE 스트림 (ChatGPT가 이 스트림에서 JSON-RPC 요청을 보냄)
# ======================================================================
async def sse_event_stream():
    # 첫 연결 이벤트 알림
    yield "data: {}\n\n".format(json.dumps({"event": "connected"}))

    while True:
        req = await client_queue.get()
        yield f"data: {json.dumps(req)}\n\n"


@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream"
    )


# ======================================================================
# ChatGPT가 JSON-RPC 요청을 보내는 엔드포인트 (POST /rpc)
# ======================================================================
@app.post("/rpc")
async def rpc_endpoint(request: Request):
    body = await request.json()

    #
    # 구조 예시:
    # { "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {} }
    #

    method = body.get("method")
    request_id = body.get("id")

    # ========== initialize ==========
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "ashen-mcp", "version": "1.0.0"},
            "serverInfo": {"name": "ashen-mcp", "version": "1.0.0"},
            "tools": [{
                "name": "hello",
                "description": "Return a greeting",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }]
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # ========== tools/list ==========
    if method == "tools/list":
        result = {
            "tools": [{
                "name": "hello",
                "description": "Return a greeting",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }
            }]
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    # ========== tools/call ==========
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "hello":
            name = arguments.get("name", "unknown")
            greeting = f"Hello, {name}!"
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": greeting}
            }

    # 알 수 없는 method
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "Method not found"}
    }
