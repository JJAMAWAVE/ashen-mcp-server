from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import Any, Dict, Optional

app = FastAPI()

# ===== SETTINGS =====
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"         # 최신 스펙 버전
BASE_URL = "https://ashen-mcp-server.onrender.com"


# ===== CORS ALLOW =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# 0. ROOT ENDPOINT — ChatGPT가 MCP URL 테스트 시 필수
# ------------------------------------------------------
@app.get("/", status_code=200)
async def root():
    return {
        "status": "ok",
        "name": SERVER_NAME,
        "version": SERVER_VERSION
    }


@app.get("/health", status_code=200)
async def health():
    return {"status": "healthy"}


# ------------------------------------------------------
# 1. MCP METADATA (.well-known/mcp.json)
# ------------------------------------------------------
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    content = {
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
        # 향후 Streamable HTTP 사용 가능 (선택)
        "streamableHttp": {
            "url": f"{BASE_URL}/mcp"
        }
    }

    response = JSONResponse(content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.head("/.well-known/mcp.json")
async def mcp_metadata_head():
    return Response(status_code=200)


# ------------------------------------------------------
# 2. SSE STREAM — ChatGPT 지속 연결
# ------------------------------------------------------
async def sse_stream():
    # 초기 연결 이벤트
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    await asyncio.sleep(0.2)

    # keep-alive
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(10)


@app.get("/sse")
async def sse_endpoint():
    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "Pragma": "no-cache"
    }
    return StreamingResponse(sse_stream(), headers=headers)


# ------------------------------------------------------
# 3. JSON-RPC HANDLER (CORE MCP LOGIC)
# ------------------------------------------------------
async def handle_rpc(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = body.get("method")
    params = body.get("params", {})
    rpc_id = body.get("id")

    is_notification = rpc_id is None

    def ok(result: Any) -> Optional[Dict[str, Any]]:
        if is_notification:
            return None
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": result
        }

    def err(code: int, message: str):
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": code, "message": message}
        }

    # INITIALIZE
    if method == "initialize":
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
        })

    # TOOL LIST
    if method == "tools/list":
        return ok({
            "tools": [
                {
                    "name": "hello",
                    "description": "Return a greeting message.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            ]
        })

    # TOOL CALL
    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "hello":
            name = args.get("name", "friend")
            return ok({
                "content": [{"type": "text", "text": f"Hello, {name}!"}],
                "isError": False
            })

        return err(-32601, f"Unknown tool: {tool_name}")

    # UNKNOWN METHOD
    if is_notification:
        return None

    return err(-32601, f"Unknown method: {method}")


@app.post("/messages")
async def messages_endpoint(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)

    response_payload = await handle_rpc(body)

    if response_payload is None:
        return Response(status_code=204)

    return JSONResponse(response_payload)


# ------------------------------------------------------
# 4. STREAMABLE HTTP (Optional)
# ------------------------------------------------------
@app.post("/mcp")
async def streamable_mcp(request: Request):
    body = await request.json()
    response = await handle_rpc(body)
    return JSONResponse(response)
