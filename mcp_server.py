from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()

# -------------------------------------------------
# CORS 설정
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MCP metadata
# -------------------------------------------------
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }

# -------------------------------------------------
# SSE 스트림
# ChatGPT MCP에서 요구하는 구조에 완벽히 맞춤
# -------------------------------------------------
@app.get("/sse")
async def sse_endpoint():

    async def event_generator():
        # 최초 연결 알림
        yield f"data: {json.dumps({'event': 'connected'})}\n\n"
        await asyncio.sleep(0.1)

        # 무한 heartbeat
        while True:
            yield f"data: {json.dumps({'event': 'alive'})}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------
# MCP: GET /tools
# -------------------------------------------------
@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {
                "name": "ping",
                "description": "Returns a pong message",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        ]
    }

# -------------------------------------------------
# MCP: POST /tools/ping
# -------------------------------------------------
@app.post("/tools/ping")
async def ping_tool(request: Request):
    body = await request.json()
    msg = body.get("message", "empty")
    return {
        "response": f"Pong: {msg}"
    }
