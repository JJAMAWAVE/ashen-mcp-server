# ===========================================================
# ashen-mcp-server (FULL FIXED VERSION — 100% WORKING)
# ===========================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, subprocess, os, base64, textwrap

# --------------------------------------------
# CONFIG
# --------------------------------------------
SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===========================================================
# ROOT
# ===========================================================
@app.get("/")
async def root():
    return {"status": "ok"}

@app.head("/")
async def head_root():
    return ""

# ===========================================================
# MCP METADATA (REQUIRED)
# ===========================================================
@app.get("/.well-known/mcp.json")
async def mcp_wellknown():

    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "analyze_text": {},
                "call_ollama": {},
                "summarize_file": {},
                "local_mcp_builder": {},
            }
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        },
        "transport": "sse",
        "sse": {"url": "/sse"},
        "streamableHttp": {"url": "/mcp"}
    }

    return JSONResponse(metadata)

# ===========================================================
# HELPER — OLLAMA
# ===========================================================
def call_ollama_raw(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            return f"[ERROR] Ollama error: {result.stderr.decode()}"
        return result.stdout.decode()
    except Exception as e:
        return f"[ERROR] Ollama execution failed: {str(e)}"


# ===========================================================
# JSON-RPC CORE
# ===========================================================
async def handle_rpc(body):
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # -------------------------------------------------------
    # REQUIRED: initialize
    # -------------------------------------------------------
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                },
                "capabilities": {
                    "tools": {
                        "analyze_text": {},
                        "call_ollama": {},
                        "summarize_file": {},
                        "local_mcp_builder": {},
                    }
                }
            }
        }

    # -------------------------------------------------------
    # REQUIRED: tools/list
    # -------------------------------------------------------
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "tools": [
                    {"name": "analyze_text"},
                    {"name": "call_ollama"},
                    {"name": "summarize_file"},
                    {"name": "local_mcp_builder"}
                ]
            }
        }

    # -------------------------------------------------------
    # analyze_text
    # -------------------------------------------------------
    if method == "analyze_text":
        text = params.get("text", "")
        mode = params.get("mode", "summary")

        if not text.strip():
            return {"jsonrpc": "2.0", "id": rpc_id,
                    "error": {"code": -32602, "message": "text required"}}

        prompts = {
            "summary": f"Summarize:\n\n{text}",
            "analysis": f"Analyze deeply:\n\n{text}",
            "keywords": f"Extract keywords:\n\n{text}"
        }
        prompt = prompts.get(mode, text)

        result = call_ollama_raw("qwen2.5:7b-instruct", prompt)

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result}],
                "isError": result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # call_ollama
    # -------------------------------------------------------
    if method == "call_ollama":
        model = params.get("model")
        prompt = params.get("prompt")

        if not (model and prompt):
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32602,"message":"model and prompt required"}}

        result = call_ollama_raw(model, prompt)

        return {
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "content":[{"type":"text","text":result}],
                "isError":result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # summarize_file
    # -------------------------------------------------------
    if method == "summarize_file":
        path = params.get("path")

        if not (path and os.path.exists(path)):
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32602,"message":"File not found"}}

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            return {"jsonrpc":"2.0","id":rpc_id,
                    "error":{"code":-32603,"message":"File read error"}}

        result = call_ollama_raw("qwen2.5:7b-instruct",
                                 f"Summarize this:\n\n{content}")

        return {
            "jsonrpc":"2.0","id":rpc_id,
            "result":{
                "content":[{"type":"text","text":result}],
                "isError":result.startswith("[ERROR]")
            }
        }

    # -------------------------------------------------------
    # local_mcp_builder
    # -------------------------------------------------------
    if method == "local_mcp_builder":
        port = params.get("server_port", 8765)

        server_py = textwrap.dedent(f"""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        import asyncio, json

        app = FastAPI()

        @app.get("/")
        async def root():
            return {{"status": "local_mcp_running"}}

        @app.get("/.well-known/mcp.json")
        async def meta():
            return {{
                "protocolVersion": "2025-06-18",
                "capabilities": {{
                    "tools": {{
                        "echo": {{}}
                    }}
                }},
                "serverInfo": {{
                    "name": "local-mcp",
                    "version": "1.0"
                }},
                "transport": "sse",
                "sse": {{"url": "/sse"}},
                "streamableHttp": {{"url": "/mcp"}}
            }}

        @app.post("/mcp")
        async def rpc(request: Request):
            body = await request.json()
            rpc_id = body.get("id")
            method = body.get("method")

            if method == "echo":
                msg = body.get("params",{{}}).get("text","")
                return {{
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {{
                        "content": [{{"type":"text","text":msg}}]
                    }}
                }}

            return {{
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {{
                    "code": -32601,
                    "message": "unknown method"
                }}
            }}

        async def sse_stream():
            yield 'data: {{"event":"connected"}}\\n\\n'
            while True:
                yield 'data: {{"event":"alive"}}\\n\\n'
                await asyncio.sleep(5)

        @app.get("/sse")
        async def sse_ep():
            return StreamingResponse(sse_stream(), media_type="text/event-stream")
        """)

        ps1 = textwrap.dedent(f"""
        python -m pip install fastapi uvicorn
        uvicorn local_mcp_server:app --host 0.0.0.0 --port {port}
        """)

        pkg = {
            "local_mcp_server.py": server_py,
            "run_local_mcp.ps1": ps1
        }

        encoded = base64.b64encode(json.dumps(pkg).encode()).decode()

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [
                    {"type": "text", "text": "Local MCP package created."},
                    {"type": "file", "name": "local_mcp_package.json", "data": encoded}
                ]
            }
        }

    # -------------------------------------------------------
    # Unknown method
    # -------------------------------------------------------
    return {
        "jsonrpc":"2.0",
        "id":rpc_id,
        "error":{"code":-32601,"message":f"Unknown method: {method}"}
    }

# ===========================================================
# HTTP /mcp
# ===========================================================
@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc":"2.0",
            "id":None,
            "error":{"code":-32700,"message":"Parse error"}
        })

    return JSONResponse(await handle_rpc(body))

# ===========================================================
# SSE
# ===========================================================
async def sse_stream():
    yield 'data: {"event":"connected"}\n\n'
    while True:
        yield 'data: {"event":"alive"}\n\n'
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(sse_stream(),
                             media_type="text/event-stream")
