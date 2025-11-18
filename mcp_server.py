##############################################
# MCP Server - Stable Plugin Version
##############################################

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import json
import pkgutil
import importlib
import plugins    # plugins 폴더 자동 로딩
##############################################

SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
BASE_URL = "https://ashen-mcp-server.onrender.com"
PROTOCOL_VERSION = "2025-06-18"

##############################################
# FastAPI 초기화
##############################################
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################################
# 플러그인 자동 로딩
##############################################
loaded_plugins = {}

for loader, module_name, ispkg in pkgutil.iter_modules(plugins.__path__):
    module = importlib.import_module(f"plugins.{module_name}")
    if hasattr(module, "run") and hasattr(module, "spec"):
        loaded_plugins[module_name] = module
        print(f"[PLUGIN] Loaded: {module_name}")

##############################################
# 1) 루트 엔드포인트 (GET + HEAD)
##############################################
@app.get("/", status_code=200)
async def root():
    return {"status": "ok", "message": "MCP server running"}

@app.head("/", status_code=200)
async def root_head():
    return ""


##############################################
# 2) MCP Metadata
##############################################
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    metadata = {
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
            "url": "/sse"
        },
        "streamableHttp": {
            "url": "/mcp"
        }
    }

    resp = JSONResponse(metadata)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


##############################################
# JSON-RPC 처리기
##############################################
async def handle_rpc(body: dict) -> dict:
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    ###################################
    # initialize
    ###################################
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }

    ###################################
    # tools/list → 플러그인 자동 등록
    ###################################
    elif method == "tools/list":
        tools = []
        for name, module in loaded_plugins.items():
            tools.append(module.spec())

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"tools": tools}
        }

    ###################################
    # tools/call → 플러그인 실행
    ###################################
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name in loaded_plugins:
            result_obj = loaded_plugins[tool_name].run(args)
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": result_obj
            }

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32601,
                "message": f"Unknown tool: {tool_name}"
            }
        }

    ###################################
    # 기타 알 수 없는 요청
    ###################################
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }


##############################################
# 3) Streamable HTTP 엔드포인트
##############################################
@app.post("/mcp")
async def rpc_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)

    result = await handle_rpc(body)

    resp = JSONResponse(result)
    resp.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return resp


##############################################
# 4) SSE 엔드포인트
##############################################
async def sse_stream():
    yield f"data: {json.dumps({'event': 'connected'})}\n\n"
    while True:
        yield f"data: {json.dumps({'event': 'alive'})}\n\n"
        await asyncio.sleep(5)

@app.get("/sse")
async def sse_endpoint():
    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
    }
    return StreamingResponse(sse_stream(), headers=headers)
