# plugins/call_ollama.py
# 임의 Ollama 모델 호출 도구

import subprocess
import json

TOOL = {
    "name": "call_ollama",
    "description": "Call any local Ollama model with a prompt.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "prompt": {"type": "string"}
        },
        "required": ["model", "prompt"]
    }
}

async def run(params):
    model = params.get("model")
    prompt = params.get("prompt")

    try:
        result = subprocess.check_output(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stderr=subprocess.STDOUT
        ).decode("utf-8")

        return {
            "content": [{"type": "text", "text": result}],
            "isError": False
        }

    except subprocess.CalledProcessError as e:
        return {
            "content": [{"type": "text", "text": f"[ERROR] Model run failure: {e.output}"}],
            "isError": True
        }
