from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

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

print("問題:", question)
print("相似度:", float(scores[0][best_index]))
print("找到的內容:")
print(best_document)
