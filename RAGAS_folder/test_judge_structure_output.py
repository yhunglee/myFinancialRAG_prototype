import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel


LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
JUDGE_MODEL = "qwen3.5-9b"


class JudgeResult(BaseModel):
    verdict: bool
    reason: str


async def main():

    client = AsyncOpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        timeout=180.0,
    )

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一個財務答案評估模型。",
            },
            {
                "role": "user",
                "content": (
                    "判斷以下陳述是否正確："
                    "台積電股票代號是 2330。"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "boolean",
                        },
                        "reason": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "verdict",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    )

    print(
        response.choices[0].message.content
    )


if __name__ == "__main__":
    asyncio.run(main())