# ============================
# TOOL: local_mcp_builder
# ============================

if method == "local_mcp_builder":
    import base64
    import textwrap
    
    include_ollama = params.get("include_ollama_setup", True)
    port = params.get("server_port", 8765)

    # 1) 로컬 MCP 서버 코드 생성
    local_server_py = textwrap.dedent(f"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import json, asyncio
    
    app = FastAPI()

    @app.get("/")
    async def root():
        return {{"status":"local_mcp_running"}}

    @app.get("/.well-known/mcp.json")
    async def meta():
        return {{
            "protocolVersion": "2025-06-18",
            "capabilities": {{"tools": {{"echo": {{"name":"echo","description":"Echo text","inputSchema":{{"type":"object","properties":{{"text":{{"type":"string"}}}},"required":["text"]}}}}}}},
            "serverInfo": {{"name":"local-mcp","version":"1.0"}},
            "transport":"sse",
            "sse":{{"url":"/sse"}},
            "streamableHttp":{{"url":"/mcp"}}
        }}

    @app.post("/mcp")
    async def rpc(request:Request):
        body = await request.json()
        method = body.get("method")
        rpc_id = body.get("id")

        if method == "echo":
            text = body.get("params",{{}}).get("text","")
            return {{"jsonrpc":"2.0","id":rpc_id,"result":{{"content":[{{"type":"text","text":text}}]}}}}

        return {{"jsonrpc":"2.0","id":rpc_id,"error":{{"code":-32601,"message":"unknown method"}}}}

    async def sse_stream():
        yield "data: {{\\"event\\": \\"connected\\"}}\\n\\n"
        while True:
            yield "data: {{\\"event\\": \\"alive\\"}}\\n\\n"
            await asyncio.sleep(5)

    @app.get("/sse")
    async def sse_ep():
        return StreamingResponse(sse_stream(), media_type="text/event-stream")
    """)

    # 2) 실행 스크립트 생성
    ps1_script = textwrap.dedent(f"""
    $env:PYTHONIOENCODING="utf-8"
    python -m pip install fastapi uvicorn

    uvicorn local_mcp_server:app --host 0.0.0.0 --port {port}
    """)

    # 3) 패키지 구성
    file_package = {
        "local_mcp_server.py": local_server_py,
        "run_local_mcp.ps1": ps1_script
    }

    encoded = base64.b64encode(json.dumps(file_package).encode()).decode()

    return {{
        "jsonrpc":"2.0",
        "id": rpc_id,
        "result": {{
            "content": [
                {{"type":"text","text": "Local MCP package created."}},
                {{"type":"file","name":"local_mcp_package.json","data": encoded}}
            ]
        }}
    }}
