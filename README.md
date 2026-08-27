# Financial Report RAG Prototype for Taiwan & US Stock Markets

## Project Overview

This project is designed as a personal portfolio and technical capability showcase.

[![Demo Video](http://img.youtube.com/vi/IZwksbgmWEw/0.jpg)](https://www.youtube.com/watch?v=IZwksbgmWEw "YouTube Demo")
![Multi-turn Conversation](./screenshots/multirounds-chat-terminal.png)

### Note

This project is under active development and is not the final version.

## Prerequisites

1. Install LM Studio and use the **Gemma4-E2B-Instruct** model. Enable the **Developer → Server API** and allow the API endpoint to be accessed via localhost, for example: `http://127.0.0.1:1234`.
2. Install NVIDIA CUDA 13.1 or 12.6.
3. Install Python 3.13.
4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Modify the last five lines of `financial_ingestion_advanced.py` to configure the local ChromaDB vector database for each company's financial report embeddings.

## Usage

Start the chat interface:

```bash
chainlit run app.py -w
```

## Features

This prototype focuses on financial statement data retrieval and does not include qualitative business analysis.

### ✅ Implemented RAG Features

1. Extracts tables from documents using fixed-size chunking, sliding windows, and OCR.
2. Rewrites user queries, supports multi-turn conversations, and converts queries into structured JSON format.
3. Normalizes stock ticker names by converting company aliases, nicknames, and informal names into official ticker symbols.
4. Uses metadata filtering to improve data relevance and source selection.
5. Supports idempotent upsert with vector database indexes using the format:
   `Market-StockTicker-Year-Quarter-Index`
6. Separates the knowledge base from conversation history using ChromaDB and in-memory storage.
7. Provides a GUI chat interface.
8. Includes a RAGAS evaluation pipeline with:
   - Faithfulness
   - Factual Correctness
   - Context Recall
   - Judge model: `qwen3.5-9b`

### Architecture Highlights

1. **Strategy Pattern for Document Parsing**

   The document parser is designed with the Strategy Pattern. This makes it easier to replace or upgrade parsing tools in the future.

   The current workflow first uses [anydoc](https://github.com/firecrawl/anydoc) for document parsing. If parsing fails, the system falls back to [docling](https://github.com/docling-project/docling), which provides OCR support.

2. **Adapter Pattern for Vector Indexing and Stock Ticker Normalization**

   The Adapter Pattern is used to handle differences between vector database index names and stock ticker names.

   The system can automatically convert queries by adding or removing market prefixes and resolving company aliases.

3. **Graceful Degradation**

   The system first attempts to convert user questions into structured JSON and resolve coreference in multi-turn conversations.

   If structured output fails, such as when the model returns Markdown or invalid JSON, the system falls back to rule-based entity extraction and vector search.

### Main Files

1. `myrag_module.py`
2. `financial_ingestion_advanced.py`
3. `entity_normalizer.py`
4. `rag6_practice.py`
5. `test_financial_rag.py`
6. `RAGAS_folder/ragas_generate_dataset.py`
7. `RAGAS_folder/evaluate_ragas.py`

### Not Yet Implemented

1. Agentic RAG
2. HyDE (Hypothetical Document Embeddings)
3. Router for distinguishing quantitative and qualitative questions
4. Hybrid search, including BM25 and re-ranking
5. LlamaIndex
6. LangChain

## System Requirements

This project can run locally and has been verified on an NVIDIA RTX 4080 GPU with Gemma4 2B and 4B large language models.

A GPU is not strictly required, but CPU-only execution will be slower.

## FAQ

1. Table documents with contextual information are parsed using **Anydoc**, which preserves context.
2. Scanned table documents are processed using **Docling** with OCR.
3. The current knowledge base only contains the 2025 Q4 quarterly reports of **TSMC** and **MediaTek**.
4. RAGAS v0.4.3 had runtime issues in this project, so it was downgraded to v0.4.1 after troubleshooting.
5. When running the RAGAS judge model through LM Studio, Qwen3.5's `enable_thinking` must be disabled. Otherwise, the evaluation may repeatedly exceed the maximum output token limit.
6. RAGAS evaluation summary (`top_k = 5` and `top_k = 10`):

```plaintext
top_k = 5
========================================
RAGAS Evaluation Summary
========================================
faithfulness             : 0.5338
factual_correctness      : 0.2112
context_recall           : 0.6587
----------------------------------------
top_k = 10
========================================
RAGAS Evaluation Summary
========================================
faithfulness             : 0.5894
factual_correctness      : 0.2142
context_recall           : 0.7320
```

## Known Issues

1. User conversation sessions are not yet fully isolated. This issue is currently not addressed.

## License

AGPL-v3

## Dependencies

1. anydoc
2. Docling

```bibtex
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
```

---

# 台美股公司財報 RAG 產品雛形

## 專案概述

本專案用於個人作品集與技術能力展示。

[![Demo 影片](http://img.youtube.com/vi/IZwksbgmWEw/0.jpg)](https://www.youtube.com/watch?v=IZwksbgmWEw "YouTube Demo")
![多輪對話](./screenshots/multirounds-chat-terminal.png)

### 提醒

本專案仍持續開發中，並非最終完成品。

## 使用前的前置作業

1. 安裝 LM Studio，並使用 **Gemma4-E2B-Instruct** 模型。啟用 **Developer → Server API**，並且令 API 入口可以透過 localhost 存取，例如：`http://127.0.0.1:1234`。
2. 安裝 NVIDIA CUDA 13.1 或 12.6。
3. 安裝 Python 3.13。
4. 安裝其他套件：

   ```bash
   pip install -r requirements.txt
   ```

5. 修改 `financial_ingestion_advanced.py` 的最後五行參數，用於建立各公司財報在本機 ChromaDB 的向量資料。

## 使用方式

啟動對談畫面：

```bash
chainlit run app.py -w
```

## 功能特點

本產品雛形聚焦於財報數據檢索，不包含質化商業分析。

### ✅ 已實作的 RAG 功能

1. 從文件中萃取表格，並使用 fixed-size chunking、sliding window 與 OCR。
2. Rewrite query，支援多輪對話，並將使用者問題轉成結構化 JSON 格式。
3. 股票代號名稱正規化，可將公司俗稱、綽號、別名轉換成正式股票代號。
4. 使用 metadata filtering 提升資料相關性與來源篩選能力。
5. 支援 idempotent upsert，向量資料庫索引格式為：
   `Market-StockTicker-Year-Quarter-Index`
6. 使用 ChromaDB 與記憶體，將知識庫資料與對話歷史隔離。
7. 提供 GUI 對話介面。
8. 建立 RAGAS 評測流程，包含：
   - Faithfulness
   - Factual Correctness
   - Context Recall
   - 裁判模型：`qwen3.5-9b`

### 架構亮點

1. **文件解析器使用 Strategy Pattern**

   文件解析器採用 Strategy Pattern 設計，未來如果有更強的解析工具，或需要改變解析策略，可以更容易抽換。

   目前流程會先使用 [anydoc](https://github.com/firecrawl/anydoc) 解析文件。如果解析失敗，則自動改用具備 OCR 機制的 [docling](https://github.com/docling-project/docling)。

2. **向量索引與股票代號正規化使用 Adapter Pattern**

   Adapter Pattern 應用在向量庫索引與股票代號名稱正規化。

   當向量庫索引名稱與股票代號不一致時，系統可以自動增減市場代號，並將公司別名轉換成查詢所需格式。

3. **Graceful Degradation**

   系統會先嘗試將使用者問題轉成結構化 JSON，並處理多輪對話中的指代與上下文關係。

   若結構化輸出失敗，例如模型回傳 Markdown 或不符合 JSON 格式的內容，系統會降級為基於規則的實體名稱判斷與向量搜尋。

### 主要檔案

1. `myrag_module.py`
2. `financial_ingestion_advanced.py`
3. `entity_normalizer.py`
4. `rag6_practice.py`
5. `test_financial_rag.py`
6. `RAGAS_folder/ragas_generate_dataset.py`
7. `RAGAS_folder/evaluate_ragas.py`

### 尚未實作

1. Agentic RAG
2. HyDE（Hypothetical Document Embeddings）
3. 判斷問題屬於量化或質化分析的 Router
4. Hybrid search，包含 BM25 與 re-ranking
5. LlamaIndex
6. LangChain

## 系統需求

本專案可在地端正常運作，已驗證可於 NVIDIA RTX 4080 顯卡搭配 Gemma4 2B 與 4B 大語言模型正常執行。

不使用顯卡也能執行，但 CPU-only 的執行速度會較慢。

## FAQ

1. 具有上下文資訊的表格文件仰賴 **Anydoc** 解析，它會保留表格情境。
2. 掃描版表格文件會改用 **Docling** 進行 OCR 處理。
3. 目前知識庫僅包含 **台積電** 與 **聯發科** 的 2025 Q4 季報。
4. RAGAS v0.4.3 官方版在本專案執行時出現問題，因此經過除錯後降級至 v0.4.1。
5. 透過 LM Studio 執行 RAGAS 裁判模型時，必須停用 Qwen3.5 的 `enable_thinking`，否則評測時容易反覆超過最大輸出 token 限制。
6. RAGAS 評估總結（`top_k = 5` 與 `top_k = 10`）：

```plaintext
top_k = 5
========================================
RAGAS Evaluation Summary
========================================
faithfulness             : 0.5338
factual_correctness      : 0.2112
context_recall           : 0.6587
----------------------------------------
top_k = 10
========================================
RAGAS Evaluation Summary
========================================
faithfulness             : 0.5894
factual_correctness      : 0.2142
context_recall           : 0.7320
```

## 已知問題

1. 每位使用者的對話訊息尚未完全隔離。此問題目前暫不處理。

## 軟體授權

AGPL-v3

## 套件

1. anydoc
2. Docling

```bibtex
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
```