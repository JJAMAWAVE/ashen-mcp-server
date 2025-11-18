from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
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

# ---------------------------------------------------------
# 📌 MCP Creator가 가장 먼저 호출하는 health check
# ---------------------------------------------------------

@app.get("/")
async def root_get():
    return {"status": "ok"}

@app.head("/")
async def root_head():
    # HEAD 요청은 body 없이 200만 주면 충분함
    return JSONResponse(content=None, status_code=200)


# ---------------------------------------------------------
# 📌 MCP Metadata (필수)
# ---------------------------------------------------------

@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }


@app.head("/.well-known/mcp.json")
async def mcp_metadata_head():
    return JSONResponse(content=None, status_code=200)


# ---------------------------------------------------------
# 📌 SSE (Server Sent Events) – MCP Creator 필수 통신 방식
# ---------------------------------------------------------

async def sse_stream():
    while True:
        yield f"data: {json.dumps({'status': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(sse_stream(), media_type="text/event-stream")


# ---------------------------------------------------------
# 📌 Tools – MCP Tool Registry
# ---------------------------------------------------------

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


@app.post("/tools/ping")
async def ping_tool(request: Request):
    body = await request.json()
    msg = body.get("message", "empty")
    return {"response": f"Pong: {msg}"}
