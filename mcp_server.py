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

# Root endpoint (ChatGPT MCP에서 필수)
@app.get("/")
async def root():
    return {"status": "MCP server running"}

# MCP metadata (.well-known)
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }

# SSE event stream
async def sse_stream():
    while True:
        yield f"data: {json.dumps({'status': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(sse_stream(), media_type="text/event-stream")

# MCP required endpoint: GET /tools
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

# MCP required endpoint: POST /tools/{tool_name}
@app.post("/tools/ping")
async def ping_tool(request: Request):
    body = await request.json()
    msg = body.get("message", "empty")
    return {"response": f"Pong: {msg}"}
