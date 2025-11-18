from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import datetime

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 콘솔 로그 함수
def log(msg: str):
    print(f"[{datetime.datetime.now().isoformat()}] {msg}", flush=True)


# ===============================
# 1) ROOT HANDLER (404 방지)
# ===============================
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body>
            <h2>ashen-mcp-server is running</h2>
            <p>Use the following endpoints:</p>
            <ul>
                <li>/.well-known/mcp.json</li>
                <li>/sse</li>
                <li>/tools</li>
            </ul>
        </body>
    </html>
    """


# ===============================
# 2) MCP METADATA
# ===============================
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    log("MCP metadata requested")
    return {
        "name": "ashen-mcp-server",
        "version": "1.0.0",
        "sse_url": "https://ashen-mcp-server.onrender.com/sse",
        "tools_url": "https://ashen-mcp-server.onrender.com/tools"
    }


# ===============================
# 3) SSE STREAM
# ===============================
async def sse_stream(request: Request):
    client_ip = request.client.host
    agent = request.headers.get("user-agent", "")

    log(f"[SSE] Client connected: {client_ip} | UA: {agent}")

    # ChatGPT MCP 접속인지 자동 감지
    if "ChatGPT" in agent or "Mozilla" not in agent:
        log("[SSE] Suspected ChatGPT MCP client detected")

    # 첫 메시지: ChatGPT 요구사항 충족
    yield "event: message\ndata: {\"status\":\"connected\"}\n\n"

    # heartbeat loop
    try:
        while True:
            await asyncio.sleep(3)

            if await request.is_disconnected():
                log(f"[SSE] Client disconnected: {client_ip}")
                break

            heartbeat_payload = {"type": "heartbeat", "msg": "alive"}
            yield f"event: message\ndata: {json.dumps(heartbeat_payload)}\n\n"

            log(f"[SSE] → heartbeat sent to {client_ip}")

    except Exception as e:
        log(f"[SSE] ERROR: {e}")


@app.get("/sse")
async def sse_endpoint(request: Request):
    return StreamingResponse(
        sse_stream(request),
        media_type="text/event-stream"
    )


# ===============================
# 4) TOOLS
# ===============================
@app.get("/tools")
async def list_tools():
    log("Tool list requested")
    return {
        "tools": [
            {
                "name": "ping",
                "description": "Returns a pong message",
                "input_schema": {
                    "type": "object",
                    "properties": { "message": { "type": "string" } },
                    "required": ["message"]
                }
            }
        ]
    }


@app.post("/tools/ping")
async def ping_tool(request: Request):
    body = await request.json()
    message = body.get("message", "empty")

    log(f"PING tool called: {message}")

    return {"response": f"Pong: {message}"}
