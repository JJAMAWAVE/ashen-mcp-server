from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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

@app.get("/.well-known/mcp.json")
async def mcp_wellknown():
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }

async def event_stream():
    while True:
        yield f"data: {json.dumps({'status': 'alive'})}\n\n"
        await asyncio.sleep(10)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {
                "name": "ping",
                "description": "테스트용 ping 툴",
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
async def run_ping(request: Request):
    body = await request.json()
    message = body.get("message", "")
    return {"response": f"Pong: {message}"}
