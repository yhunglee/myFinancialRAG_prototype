from __future__ import annotations

"""接上 LM studio
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import requests

def ask_local_llm(
    question: str,
    context: str,
) -> str:
  prompt = f"""
    你是一個文件問答助理。

    請只根據下面提供的參考資料回答問題。
    如果參考資料不足，請直接回答「根據現有資料無法判斷」。
    不要使用參考資料以外的知識補充答案。

    參考資料:
    ---
    {context}
    ---

    問題:
    {question}
    """
  response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={
      "Context-Type": "application/json",
    },
    json={
      # 使用 LM Studio API 顯示的模型名稱
      "model": "gemma-4-e2b-it",
      "messages": [
        {
          "role": "user",
          "content": prompt,
        }
      ],
      "temperature": 0.1,
    },
    timeout=120,
  )

  response.raise_for_status()
  result = response.json()
  return result["choices"][0]["message"]["content"]

documents = [
    """
    React reconciliation 是 React 比較新舊元素樹，
    並決定哪些畫面需要更新的過程。
    """,
    """
    在同一層的清單中，key 用來協助 React 識別元素身分。
    key 不應使用會頻繁改變的隨機值。
    """,
    """
    useEffect cleanup 會在元件卸載前，
    以及下一次 effect 執行前被呼叫。
    """,
]

model = SentenceTransformer(
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

document_embeddings = model.encode(
  documents,
  normalize_embeddings=True,
)

question = "React 為什麼要求清單項目提供 key？"

question_embedding = model.encode(
  [question],
  normalize_embeddings=True,
)

# normalize 後，內積等同 cosine similarity
scores = question_embedding @document_embeddings.T

best_index = int(np.argmax(scores))
best_document = documents[best_index]


answer = ask_local_llm(
  question=question,
  context=best_document,
)

print("回答:")
print(answer)