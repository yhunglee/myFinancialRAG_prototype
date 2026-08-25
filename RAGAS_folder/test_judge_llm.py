from __future__ import annotations

from openai import OpenAI

client = OpenAI(
  base_url="http://127.0.0.1:1234/v1",
  api_key="lm-studio", # 任意填，不要空字串即可
)


# MODEL_NAME = "gemma-4-e2b-it"

# response = client.chat.completions.create(
#   model=MODEL_NAME,
#   messages=[
#     {
#         "role": "system",
#         "content": "你是一名財務資料問答系統的評估員。"
#     },
#     {
#         "role": "user",
#         "content": "台積電第四季營收是 1000 億。這句話是財務事實陳述嗎？"
#     }
#   ],
#   temperature=0,
# )

# print(response.choices[0].message.content)

models = client.models.list()

print("LM Studio models:")
for model in models.data:
  print(model.id)

