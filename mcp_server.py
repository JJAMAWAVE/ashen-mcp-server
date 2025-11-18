# ================================
#  MCP SERVER (FINAL STABLE EDITION)
#  플러그인 자동 로더 + JSON-RPC + SSE
#  대장님은 plugins 폴더만 수정하면 됩니다
# ================================

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import importlib
import pkgutil
import json
import os

SERVER_NAME = "ashen-mcp-server"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2025-06-18"
BASE_URL = "https://ashen-mcp-server.onrender.com"

app = FastAPI()

# ==================================
# CORS
# ==================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================
# 1) ROOT (GET / HEAD)
# ==================================
@app.get("/", status_code=200)
async def root():
    return {"status": "ok", "message": "MCP server running"}

@app.head("/", status_code=200)
async def root_head():
    return ""


# ==================================
# 2) 플러그인 자동 로딩 시스템
# ==================================

PLUGIN_FOLDER = "plugins"
loaded_plugins = {}   # name → module
tool_definitions = {} # name → TOOL dict

def load_plugins():
    """
    plugins 폴더 안의 모든 *.py 파일을 자동 로딩한다.
    TOOL 와 run() 함수가 있으면 자동 등록
    """
    global loaded_plugins, tool_definitions

    if not os.path.isdir(PLUGIN_FOLDER):
        print("[WARN] plugins 폴더 없음, 생성 중…")
        os.makedirs(PLUGIN_FOLDER, exist_ok=True)

    print("=== MCP Plugin Loader ===")
    for finder, name, ispkg in pkgutil.iter_modules([PLUGIN_FOLDER]):
        module_path = f"{PLUGIN_FOLDER}.{name}"
        print(f" - Loaded plugin: {module_path}")

        module = importlib.import_module(module_path)

        if hasattr(module, "TOOL") and hasattr(module, "run"):
            tool_name = module.TOOL.get("name")
            tool_definitions[tool_name] = module.TOOL
            loaded_plugins[tool_name] = module

        else:
            print(f"   [WARN] {name}.py: TOOL or run() 없음 → 스킵")

    print("=== Plugin Loading Complete ===")


# 서버 시작 시 플러그인 자동 로딩
load_plugins()


# ==================================
# 3) MCP Metadata
# ==================================
@app.get("/.well-known/mcp.json")
async def mcp_metadata():
    metadata = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": tool_definitions
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


# ==================================
# 4) JSON-RPC 메서드 처리기
# ==================================
async def dispatch_rpc(body: dict):
    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    # initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": tool_definitions
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }

    # plugin tool
    if method in loaded_plugins:
        try:
            result = await loaded_plugins[method].run(params)
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32000,
                    "message": f"Tool execution error: {e}"
                }
            }

    # unknown method
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }


# ==================================
# 5) Streamable HTTP
# ==================================
@app.post("/mcp")
async def mcp_http(request: Request):
    try:
        body = await request.json()
    except:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }, status_code=400)

    result = await dispatch_rpc(body)

    resp = JSONResponse(result)
    resp.headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
    return resp


# ==================================
# 6) SSE Transport
# ==================================
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
