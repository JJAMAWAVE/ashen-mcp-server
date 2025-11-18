# plugins/analyze_text.py
# 텍스트 분석/요약 (기본 모델: qwen2.5:7b)

import subprocess
import json

MODEL = "qwen2.5:7b"

TOOL = {
    "name": "analyze_text",
    "description": "Analyze or summarize text using offline Ollama models.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["summary", "analysis", "keywords"],
                "default": "summary"
            }
        },
        "required": ["text"]
    }
}

async def run(params):
    text = params.get("text")
    mode = params.get("mode", "summary")

    if not text:
        return {
            "content": [{"type": "text", "text": "No text provided."}],
            "isError": True
        }

    prompt = f"""
당신은 전문 분석가 AI 입니다.
작업 모드: {mode}

분석할 텍스트:
{text}

요약 또는 분석 결과만 간결히 출력하십시오.
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
            "content": [{"type": "text", "text": f"[ERROR] Ollama execution failed: {e.output}"}],
            "isError": True
        }
