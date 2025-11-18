from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP metadata (.well-known)
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }

# SSE with immediate first event
async def sse_stream():
    # 🔥 ChatGPT가 요구하는 즉시 첫 메시지
    yield f"data: {json.dumps({'status': 'connected'})}\n\n"

    # 이후 heartbeat
    while True:
        await asyncio.sleep(3)
        yield f"data: {json.dumps({'status': 'alive'})}\n\n"

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(sse_stream(), media_type="text/event-stream")


# Tools list
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

# Tool method
@app.post("/tools/ping")
async def ping_tool(request: Request):
    body = await request.json()
    msg = body.get("message", "empty")
    return {"response": f"Pong: {msg}"}
