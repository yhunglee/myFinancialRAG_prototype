from __future__ import annotations
"""多輪對話RAG 範例"""

from myrag_module import FinancialRAGService


if __name__ == "__main__":

  rag_service = FinancialRAGService(
    db_path="./chroma_db",
    collection_name="financial_reports"
  )

  print("=== Financial RAG Service 啟動 ===")

  # 模擬多輪回答
  q1 = "台積電 2025 年第四季營收是多少？"
  print(f"User: {q1}")
  print(f"AI: {rag_service.rag_chat(q1)}\n")

  # q2 = "那它的營業利益率是多少?"
  q2 = "台積電和發哥的 2025 年第四季營收是多少?"
  print(f"User: {q2}")
  print(f"AI: {rag_service.rag_chat(q2, top_k=5)}\n")