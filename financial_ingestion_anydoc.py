from __future__ import annotations

from pathlib import Path
from typing import Any
import anydoc
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class FinancialIngestionPipeline:
  """以 anydoc 為核心的台美股財報解析、Markdown 切塊與向量資料庫載入管道"""

  def __init__(
      self,
      db_path: str = "./chroma_db",
      collection_name: str = "financial_reports",
      embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      chunk_size: int = 600,
      chunk_overlap: int = 120,
  ) -> None:
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    # 1. 啟用 RTX 4080(CUDA) 加速
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[系統資訊] Embedding 模型運作裝置: {self.device}")

    self.embedding_model = SentenceTransformer(
      embedding_model_name,
      device=self.device
    )

    # 2. 連線持久化 ChromaDB
    self.chroma_client = chromadb.PersistentClient(path=db_path)
    self.collection = self.chroma_client.get_or_create_collection(
      name=collection_name
    )


  # def extract_with_anydoc(self, pdf_path: str | Path) -> list[dict[str, Any]]:
  #   """ anydoc 解析 PDF，取得結構化 Markdown 與頁碼資訊"""

  #   pdf_path = Path(pdf_path)

  #   if not pdf_path.exists():
  #     raise FileNotFoundError(f"找不到檔案: {pdf_path}")

  #   # 使用 anydoc 解析文件版面與表格
  #   parsed_doc = anydoc.parse(str(pdf_path))
  #   pages_data: list[dict[str, Any]] = []

  #   # anydoc 支援逐頁或按結構段落(Blocks/Pages)讀取
  #   if hasattr(parsed_doc, "pages") and parsed_doc.pages:
  #     for page_idx, page in enumerate(parsed_doc.pages):
  #       # 轉為 Markdown 格式，保留財報表格的 | 欄位 | 數值 | 結構
  #       page_markdown = page.to_markdown() if hasattr(page, 'to_markdown') else str(page)
  #       cleaned_content = page_markdown.strip()
  #       if cleaned_content:
  #         pages_data.append({
  #           "page_number": getattr(page, "page_number", page_idx),
  #           "text": cleaned_content,
  #         })
  #   else:
  #     # 單一全文字 Markdown 模式
  #     full_markdown = parsed_doc.to_markdown() if hasattr(parsed_doc, "to_markdown") else str(parsed_doc)
  #     pages_data.append({
  #       "page_number": 1,
  #       "text": full_markdown.strip()
  #     })

  #   return pages_data

  def extract_full_markdown(self, pdf_path: str | Path) -> str:
    """使用 anydoc 將整份 PDF 直接轉成乾淨的 Markdown 格式"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
      raise FileNotFoundError(f"找不到檔案: {pdf_path}")

    # anydoc 核心優勢: 將各種文件急速轉為單一 Markdown，保留表格與結構
    markdown_text = anydoc.to_markdown(str(pdf_path))
    return markdown_text.strip()

  def split_markdown_into_chunks(self, text: str) -> list[str]:
    """滑動視窗切塊: 保留上下文並避免斷句"""
    if len(text) <= self.chunk_size:
      return [text]

    chunks: list[str] = []
    start = 0
    step = self.chunk_size - self.chunk_overlap

    while start < len(text):
      end = start + self.chunk_size
      chunk = text[start:end]
      chunks.append(chunk)
      if end >= len(text):
        break
      start += step

    return chunks

  def ingest_pdf(
      self,
      pdf_path: str | Path,
      ticker: str,
      market: str,
      year: int,
      quarter: str,
      report_type: str = "10-Q",
      batch_size: int = 64,
  ) -> int:
    """解析財報 PDF 並批次寫入 ChromaDB"""
    print(f"\n[開始解析] {ticker} ({market}) - {year} {quarter}")
    pages_data = self.extract_full_markdown(pdf_path)

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for page_info in pages_data:
      page_num = page_info["page_number"]
      page_chunks = self.split_markdown_into_chunks(page_info["text"])

      for chunk_idx, chunk in enumerate(page_chunks):
        # 建立具業務意義的唯一 ID(防重複寫入)
        chunk_id = f"{market.upper()}-{ticker.upper()}-{year}-{quarter.upper()}-P{page_num}-C{chunk_idx}"

        # 建立結構化 Metadata 供檢索過濾
        metadata = {
          "ticker": ticker.upper(),
          "market": market.upper(),
          "year": int(year),
          "quarter": quarter.upper(),
          "report_type": report_type,
          "page_number": int(page_num),
          "chunk_index": int(chunk_idx),
          "source_file": Path(pdf_path).name,
        }

        documents.append(chunk)
        metadatas.append(metadata)
        ids.append(chunk_id)

    total_chunks = len(documents)
    print(f"-> 解析完成: 共{len(pages_data)} 頁，生成 {total_chunks}個 Chunks")

    if total_chunks == 0:
      print("-> 未取得文字，跳過寫入。")
      return 0

    # GPU 批次向量化並寫入
    print("-> 正在寫入 {len(documents)} 筆 Chunks 至 ChromaDB...")
    for i in tqdm(range(0, total_chunks, batch_size), desc="Upserting Batches"):
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

    print(f"[完成] 成功寫入 {total_chunks} 筆資料到 ChromaDB\n")
    return total_chunks

if __name__ == "__main__":
  pipeline = FinancialIngestionPipeline(
    db_path="./chroma_db",
    collection_name="financial_reports"
  )

  # 執行範例（請確保路徑下有對應 PDF 檔案）
  # pipeline.ingest_pdf(
  #     pdf_path="./2330_2025_Q4.pdf",
  #     ticker="2330",
  #     market="TW",
  #     year=2025,
  #     quarter="Q4",
  #     report_type="財務報告書"
  # )
  pipeline.ingest_pdf(
    pdf_path="./2330_25Q4.pdf",
    ticker="2330",
    market="TW",
    year=2025,
    quarter="Q4",
    report_type="財務報告"
  )