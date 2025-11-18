# plugins/analyze_text.py

import subprocess
import json

MODEL = "qwen2.5:7b-instruct"


def run_ollama(prompt: str) -> str:
    """
    Ollama 모델에게 프롬프트를 보내고 결과 텍스트를 반환하는 함수
    """
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return result.stdout.decode("utf-8")
    except Exception as e:
        return f"[ERROR] Ollama 실행 실패: {e}"


def run(args: dict) -> dict:
    """
    MCP Server → 플러그인 → ChatGPT로 응답 객체 반환하는 함수
    """
    text = args.get("text", "")
    task = args.get("task", "summarize")

    if not text:
        return {
            "content": [{"type": "text", "text": "No text provided."}],
            "isError": True,
        }

    # 시스템 프롬프트 구성
    prompt = f"""
You are a world-class document analyzer. Perform the task requested.

Task: {task}
Text:
{text}
"""

    response = run_ollama(prompt)

    return {
        "content": [{"type": "text", "text": response}],
        "isError": False,
    }


# MCP server에서 사용하는 Tool 메타데이터
def spec():
    return {
        "name": "analyze_text",
        "description": "Analyze or summarize text using offline AI models (Ollama).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze or summarize."},
                "task": {
                    "type": "string",
                    "description": "Action: summarize, analyze, rewrite, extract, etc.",
                    "default": "summarize"
                }
            },
            "required": ["text"]
        }
    }
