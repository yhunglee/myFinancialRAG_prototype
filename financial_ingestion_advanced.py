from __future__ import annotations



"""
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import anydoc
import chromadb
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm


# ==========================
# 模組一: 萃取器介面與具體實作(策略模式)
# ==========================

class DocumentExtractor(ABC):
  """定義文件萃取器的標準介面"""

  @abstractmethod
  def extract(self, pdf_path: str | Path) -> str:
    """所有萃取器都必須實作此方法，並回傳 Markdown 字串"""
    pass

class AnydocExtractor(DocumentExtractor):
  """策略 A: 使用 firecrawl-anydoc (極速、輕量)"""
  def extract(self, pdf_path: str | Path) -> str:
    print("[Anydoc] 嘗試解析 Markdown...")
    text = anydoc.to_markdown(str(pdf_path))
    return text.strip()

class DoclingExtractor(DocumentExtractor):
  """策略 B: 使用 IBM Docling (重量級、含 OCR 與版面分析)"""
  def __init__(self):
    print("[Docling] 正在初始化多模態版面分析模型到 RTX 4080")
    self.converter = DocumentConverter()

  def extract(self, pdf_path: str | Path) -> str:
    print("[Docling] 啟動深層 OCR 與表格解析...")
    result = self.converter.convert(str(pdf_path))

    # 將解析結果匯出為 Markdown
    return result.document.export_to_markdown().strip()

class FallbackExtractor(DocumentExtractor):
  """責任鏈管理器: 管理萃取器的優先順序與降級機制"""

  def __init__(self, extractors: list[DocumentExtractor], min_chars: int = 100):
    self.extractors = extractors
    # 若萃取出來的字數低於此值，視為[解析失敗](例如掃描檔只解析出空白)
    self.min_chars = min_chars

  def extract(self, pdf_path: str | Path) -> str:
    for idx, extractor in enumerate(self.extractors):
      try:
        # 取得該萃取器類別的名稱，方便印出 log
        ext_name = extractor.__class__.__name__

        text = extractor.extract(pdf_path)

        # 檢查萃取結果是否有效
        if len(text) >= self.min_chars:
          print(f"✅ [{ext_name}]萃取成功! 共{len(text)} 字。")
          return text
        else:
          print(f"⚠️ [{ext_name}] 萃取字數過少 ({len(text)} 字)，判斷為失效。")

      except Exception as e:
        print(f"❌ [{ext_name}] 發生錯誤: {e}")

      # 若不是最後一個萃取器，提示切換
      if idx < len(self.extractors) - 1:
        print("🔄 啟動降級機制，切換至下一個萃取器...\n")

    raise RuntimeError(f"所有萃取模組皆無法解析此檔案: {pdf_path}")


class FinancialIngestionPipeline:
  """財報載入管線(只負責切塊和寫入資料庫，不干涉如何萃取)"""

  def __init__(
      self,
      extractor: DocumentExtractor, # 萃取器
      db_path: str = "./chroma_db",
      collection_name: str = "financial_reports",
      chunk_size: int = 600,
      chunk_overlap: int = 150,
  ) -> None:
    self.extractor = extractor
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.embedding_model = SentenceTransformer(
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      device=self.device
    )

    self.chroma_client = chromadb.PersistentClient(
      path=db_path
    )
    self.collection = self.chroma_client.get_or_create_collection(
      name=collection_name
    )

  def split_markdown(self, text: str) -> list[str]:
    if not text: return []
    if len(text) <= self.chunk_size: return [text]

    chunks = []
    start = 0
    step = self.chunk_size - self.chunk_overlap
    while start < len(text):
      end = start + self.chunk_size
      chunks.append(text[start:end])
      if end >= len(text): break
      start += step

    return chunks

  def ingest_pdf(
      self,
      pdf_path: str | Path,
      ticker: str,
      market: str,
      year: int,
      quarter: str,
      batch_size: int = 64,
  ) -> int:
    print(f"\n========================================")
    print(f"[啟動管線] 目標: {ticker} ({market}) - {year} {quarter}")
    print(f"========================================")

    # 1. 呼叫注入的萃取器(anydoc or docling)
    start_time = time.time()
    full_markdown = self.extractor.extract(pdf_path)
    print(f"[萃取耗時] {time.time() - start_time:.2f} 秒\n")

    # 2. 進行切塊
    chunks = self.split_markdown(full_markdown)

    documents, metadatas, ids = [], [], []
    for chunk_idx, chunk in enumerate(chunks):
      if not chunk.strip(): continue

      chunk_id = f"{market.upper()}-{ticker.upper()}-{year}-{quarter.upper()}-C{chunk_idx}"
      metadata = {
        "ticker": ticker.upper(),
        "market": market.upper(),
        "year": int(year),
        "quarter": quarter.upper(),
        "chunk_index": int(chunk_idx),
      }
      documents.append(chunk)
      metadatas.append(metadata)
      ids.append(chunk_id)

    if not documents:
      print("⚠️ 未生成任何有效 Chunk，略過寫入。")
      return 0
    
    # GPU 批次向量化並寫入
    print(f"-> 正在寫入 {len(documents)} 筆 Chunks 至 ChromaDB...")
    for i in tqdm(range(0, len(documents), batch_size), desc="Upserting Batches"):
      batch_docs = documents[i : i+batch_size]
      batch_metadatas = metadatas[i: i+batch_size]
      batch_ids = ids[i: i+batch_size]

      # 先由 RTX 4080 完成向量編碼運算
      batch_embeddings = self.embedding_model.encode(
        batch_docs,
        normalize_embeddings=True,
        show_progress_bar=False,
      ).tolist()

      # 向量計算完成後，再一次性交付 ChromaDB 寫入硬碟
      self.collection.upsert(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
      )
      
    print(f"✨ [管線執行成功] 已將 {len(documents)} 筆資料 Chunk 寫入 ChromaDB 知識庫")
    return len(documents)

if __name__ == '__main__':
  # 1. 初始化各種萃取策略
  anydoc_strategy = AnydocExtractor()
  docling_strategy = DoclingExtractor()

  # 2. 建立降級責任鍊(優先試 Anydoc，若字數 < 100 或報錯，則切換到 Docling)
  smart_extractor = FallbackExtractor(
    extractors=[anydoc_strategy, docling_strategy],
    min_chars=100
  )

  # 3. 將智慧萃取器注入管線
  pipeline = FinancialIngestionPipeline(extractor=smart_extractor)

  # 4. 執行匯入
  pipeline.ingest_pdf(
    "./2330_25Q4.pdf", 
    "2330",
    "TW", 
    2025,
    "Q4"
  )
