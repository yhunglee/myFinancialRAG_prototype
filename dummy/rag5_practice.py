from __future__ import annotations
"""
搜尋知識庫, Retrieve 檢索端 + 模型推論（Generation, 串接 LLM 端)生成回答
"""

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(
  path="./chroma_db"
)

collection = client.get_or_create_collection(
  name="react_notes",
)

llm_client = OpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio"
)

question = "為什麼 React 清單需要 key？"

question_embedding = embedding_model.encode(
  [question],
  normalize_embeddings=True,
).tolist()

results = collection.query(
  query_embeddings=question_embedding,
  n_results=2,
)

retrieved_documents = results["documents"][0]
# retrieved_metadatas = results["metadatas"][0]
# retrieved_distances = results["distances"][0]


context_text = "\n".join([f"- {doc}" for doc in retrieved_documents])

system_prompt = (
  "你是一個專業的技術助理。請嚴格依據以下提供的參考資料回答問題。"
  "如果參考資料不足以回答，請誠實說明不知道，不要自行捏造。"
)

user_prompt = f"""[參考資料]
{context_text}

[問題]
{question}
"""

response = llm_client.chat.completions.create(
  model="gemma-4-e2b-it",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ],
  temperature=0.2,
)

print("=== 模型回答 ===")
print(response.choices[0].message.content)