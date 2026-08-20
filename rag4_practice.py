from __future__ import annotations
"""
搜尋知識庫, Retrieve 檢索端而已
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
retrieved_metadatas = results["metadatas"][0]
retrieved_distances = results["distances"][0]

for document, metadata, distance in zip(
  retrieved_documents,
  retrieved_metadatas,
  retrieved_distances,
):
  print("內容:", document)
  print("來源:", metadata)
  print("距離:", distance)
  print("----")