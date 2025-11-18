# plugins/summarize_file.py
# 파일 내용 읽고 요약 (기본 모델: qwen2.5:7b)

import subprocess
import os

MODEL = "qwen2.5:7b"

TOOL = {
    "name": "summarize_file",
    "description": "Read a local file and summarize its content using Ollama.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        },
        "required": ["path"]
    }
}

async def run(params):
    path = params.get("path")

    if not os.path.exists(path):
        return {
            "content": [{"type": "text", "text": f"File not found: {path}"}],
            "isError": True
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except:
        return {
            "content": [{"type": "text", "text": "Failed to read file."}],
            "isError": True
        }

    prompt = f"""
다음 문서를 요약하십시오:

{text}
"""

    try:
        result = subprocess.check_output(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            stderr=subprocess.STDOUT
        ).decode("utf-8")

        return {
            "content": [{"type": "text", "text": result}],
            "isError": False
        }

    except subprocess.CalledProcessError as e:
        return {
            "content": [{"type": "text", "text": f"[ERROR] Ollama run failed: {e.output}"}],
            "isError": True
        }
