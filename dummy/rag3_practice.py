from __future__ import annotations
"""
建立知識庫
"""

import chromadb
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

documents = [
  "React reconciliation 是 React 比較新舊元素樹並決定畫面更新的過程。",
  "在同一層清單中，key 用來協助 React 識別元素身分。",
  "useEffect cleanup 會在元素 unmount 前以及下一次 effect 執行前，呼叫。"
]

ids = [
  "react-001",
  "react-002",
  "react-003",
]

metadatas = [
  {
    "source": "react-notes",
    "topic": "reconciliation"
  },
  {
    "source": "react-notes",
    "topic": "key"
  },
  {
    "source": "react-notes",
    "topic": "useEffect"
  }
]

embeddings = embedding_model.encode(
  documents,
  normalize_embeddings=True,
).tolist()

collection.upsert(
  ids=ids,
  documents=documents,
  embeddings=embeddings,
  metadatas=metadatas,
)

print("文件已寫入向量資料庫")