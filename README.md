# Financial Report RAG Prototype for Taiwan and U.S. Equities

## Project Overview

This project is an independently developed prototype for portfolio presentation and technical capability demonstration.

It explores local Retrieval-Augmented Generation (RAG) and Agentic RAG workflows for retrieving and comparing information from corporate financial reports across Taiwan and U.S. equity markets.

[![Demo Video](http://img.youtube.com/vi/IZwksbgmWEw/0.jpg)](https://www.youtube.com/watch?v=IZwksbgmWEw "YouTube Demo")

![Multi-turn Conversation](./screenshots/multirounds-chat-terminal.png)

### Notice

This project is under active development and is not a production-ready system.

## Prerequisites

1. Install **LM Studio** and load a compatible local instruction model such as **Gemma4-E2B-Instruct**. Enable **Developer → Server API** and make the OpenAI-compatible API endpoint accessible through localhost, for example:

   ```text
   http://127.0.0.1:1234
   ```

2. Install NVIDIA CUDA 13.1 or 12.6 if GPU acceleration is required.

3. Install Python 3.13.

4. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure the `pipeline.ingest_pdf(...)` parameters at the end of `financial_ingestion_advanced.py` for each financial report before building the local ChromaDB knowledge base.

## Usage

Start the Chainlit interface:

```bash
chainlit run app.py -w
```

## Current Scope

The current prototype focuses primarily on **financial-report retrieval, evidence validation, numerical comparison, and multi-turn financial Q&A**.

It does not yet attempt to provide comprehensive qualitative investment research, valuation recommendations, or investment advice.

## Features

### ✅ Core RAG Pipeline

1. **Financial document ingestion**

   * Extracts financial-report content and tables into Markdown.
   * Supports fixed-size chunking with overlapping sliding windows.
   * Supports OCR-based fallback for scanned or layout-heavy documents.

2. **Query processing and multi-turn conversation**

   * Rewrites or contextualizes follow-up questions into standalone queries.
   * Resolves conversational references such as companies, periods, and previously discussed subjects.
   * Uses structured outputs where appropriate for downstream processing.

3. **Stock entity normalization**

   * Converts company names, aliases, nicknames, and stock symbols into canonical ticker representations.
   * Supports Taiwan and U.S. equity naming conventions.

4. **Metadata-aware retrieval**

   * Uses metadata filters such as ticker, year, and quarter to constrain vector retrieval to the intended reporting scope.

5. **Idempotent vector upsert**

   * Uses deterministic vector IDs based on:

   ```text
   Market-StockTicker-Year-Quarter-ChunkIndex
   ```

   This allows repeated ingestion without creating duplicate vector records for the same chunks.

6. **Separation of knowledge-base data and conversational state**

   * ChromaDB stores the persistent financial-report knowledge base.
   * Conversation state is managed separately by the application layer.

7. **Chainlit GUI**

   * Provides an interactive browser-based chat interface.
   * Displays Agentic RAG execution stages in collapsible steps.
   * Supports multi-turn conversations.

### ✅ RAG Evaluation with RAGAS

A local RAGAS evaluation workflow has been implemented with the following metrics:

* Faithfulness
* Factual Correctness
* Context Recall

The current local judge model is:

```text
qwen3.5-9b
```

Evaluation datasets, intermediate retrieval results, and evaluation outputs are stored under:

```text
RAGAS_folder/
```

### ✅ Agentic RAG with LangGraph

The `agentic-flow` branch extends the original RAG pipeline into a stateful Agentic RAG workflow implemented with LangGraph.

Current workflow:

```text
User Question
    ↓
Contextualize Question
    ↓
Intent Router
    ↓
Research Planner
    ↓
Evidence Reuse Checker
    ↓
RAG Executor
    ↓
Evidence Checker
    ↓
┌─────────────────────────────────────┐
│ proceed                             │
│ retrieve again                      │
│ regenerate unsupported answer       │
│ stop after retry limit              │
└─────────────────────────────────────┘
    ↓
Calculator
    ↓
Report Writer
    ↓
Final Answer
```

Implemented Agentic RAG capabilities include:

1. **Conversation contextualization**

   * Converts follow-up questions into standalone research questions.
   * Preserves company, reporting period, metric, and comparison constraints from previous turns when appropriate.

2. **Intent routing**

   * Classifies questions into:

     * `single_company`
     * `peer_comparison`
     * `industry_research`
     * `general_financial_question`

3. **Automatic research planning**

   * Decomposes a financial research request into independent retrieval tasks.
   * Separates multi-company comparisons into company-level research tasks.

4. **RAG task execution**

   * Executes generated research tasks against the existing financial-report RAG pipeline.
   * Applies company and reporting-period metadata filters before vector retrieval.

5. **Evidence validation**

   * Evaluates whether retrieved evidence is sufficient to support the requested answer.
   * Identifies missing or weak evidence.
   * Prevents unsupported generated answers from directly reaching the final report.

6. **Controlled retry loops**

   * Supports additional retrieval when evidence is insufficient.
   * Supports answer regeneration when an answer is not supported by retrieved evidence.
   * Applies retry limits to prevent infinite Agentic loops.

7. **Validated evidence reuse**

   * Reuses evidence that passed validation in previous conversation turns when company, period, and topic still match the new research task.
   * Avoids unnecessary repeated vector retrieval for suitable follow-up questions.

8. **Deterministic calculator tool**

   * Detects when a user explicitly requests a numerical difference.
   * Uses the LLM only to identify calculation operands.
   * Performs unit normalization and arithmetic in Python rather than relying on LLM arithmetic.
   * Current supported operation:

   ```text
   difference
   ```

9. **Report generation**

   * Produces the final response only after the evidence-validation and calculation stages have completed.

10. **Agent execution visibility in Chainlit**

    * Displays major LangGraph nodes and intermediate results through nested/collapsible execution steps.
    * Keeps the final answer readable while still allowing the execution process to be inspected.

## Architecture Highlights

### 1. Document Parsing with Strategy Pattern and Graceful Fallback

Financial document extraction is abstracted behind a common `DocumentExtractor` interface.

The current strategy is:

```text
Anydoc
   ↓ parsing failure / insufficient extracted content
Docling
```

The system first attempts document extraction using [Anydoc](https://github.com/firecrawl/anydoc).

If parsing fails or the extracted content is insufficient, the pipeline automatically falls back to [Docling](https://github.com/docling-project/docling), which provides OCR and layout-aware document processing.

This design allows additional document parsers to be introduced without tightly coupling them to the ingestion pipeline.

### 2. Adapter-Style Normalization for Tickers and Database Metadata

The retrieval pipeline separates user-facing stock representations from the ticker format stored in the vector database.

Entity normalization supports:

* ticker symbols
* formal company names
* aliases
* commonly used company nicknames
* market-specific ticker representations

The retrieval layer adapts normalized company entities into the metadata format required by ChromaDB.

### 3. Graceful Degradation in Query Processing

The non-Agentic RAG pipeline first attempts LLM-based structured query decomposition.

If structured parsing fails—for example, because the local model returns Markdown or invalid JSON—the system can fall back to simpler query rewriting, entity normalization, and vector retrieval instead of immediately terminating the request.

### 4. Evidence-Gated Agentic Workflow

The Agentic RAG workflow does not directly trust the first generated answer.

Retrieved evidence is checked before the system decides whether to:

```text
proceed
retrieve_again
regenerate_answer
stop
```

Both retrieval and regeneration paths have bounded retry counts to prevent infinite execution loops.

### 5. LLM Planning + Deterministic Execution

For numerical operations, the LLM is responsible for understanding user intent and identifying operands, while Python performs unit normalization and arithmetic.

This separates probabilistic language understanding from deterministic computation.

## Main Files

### Core RAG

* `myrag_module.py`

  * Core financial RAG service
  * Vector retrieval
  * Metadata filtering
  * Query decomposition and rewriting
  * LM Studio integration

* `financial_ingestion_advanced.py`

  * Financial-report ingestion pipeline
  * Anydoc / Docling fallback
  * Chunking
  * Embedding generation
  * ChromaDB upsert

* `entity_normalizer.py`

  * Company-name and ticker normalization

* `rag_event.py`

  * Structured RAG event representation

* `rag6_practice.py`

  * RAG execution example

### Agentic RAG

* `agent_state.py`

  * LangGraph shared state definition

* `agent_node.py`

  * Agent nodes
  * Research planning
  * Evidence reuse
  * Evidence validation
  * Retry logic
  * Calculator
  * Report writer

* `agent_graph.py`

  * LangGraph graph definition
  * Conditional routing
  * Retry loops

* `app.py`

  * Chainlit Agentic RAG user interface
  * Conversation history
  * Validated evidence persistence across turns
  * Agent execution-step visualization

### Evaluation

* `RAGAS_folder/ragas_generate_dataset.py`
* `RAGAS_folder/evaluate_ragas.py`
* `RAGAS_folder/base_dataset.json`

### Tests

The project also contains dedicated tests for major Agentic workflow components, including:

* intent routing
* Agent graph execution
* answer regeneration loops
* retrieval retry loops
* evidence reuse
* calculator behavior

## Planned Enhancements

The following capabilities are not yet implemented or are outside the current MVP scope:

1. **HyDE — Hypothetical Document Embeddings**

2. **Quantitative vs. qualitative research routing**

   * Explicitly classify questions that require numerical retrieval versus qualitative business analysis.

3. **Hybrid retrieval**

   * Dense vector retrieval
   * BM25 / lexical retrieval
   * Re-ranking

4. **Financial terminology normalization**

   * Centralized normalization for financial metrics, aliases, abbreviations, and equivalent terminology.

5. **Additional deterministic financial tools**

   * Ratio calculations
   * Growth-rate calculations
   * Percentage comparisons
   * Multi-company numerical analysis

6. **More advanced qualitative research**

   * Fundamental analysis
   * Management outlook analysis
   * Business and industry interpretation

7. **Multi-hop Agentic RAG evaluation**

   * Evaluate retrieval and answer quality for questions requiring multiple pieces of evidence.

8. **LangGraph-native fan-out / fan-in**

   * Execute independent research tasks as graph-level parallel branches and aggregate their results.

9. **Broader financial-report coverage**

   * Additional companies
   * Additional quarters
   * Additional Taiwan and U.S. equities

## System Requirements

The project is designed to run locally.

The current development environment has been validated on a laptop equipped with:

* NVIDIA RTX 4080 Laptop GPU
* 64 GB system memory
* Python 3.13
* LM Studio
* Local LLM inference
* ChromaDB

The prototype has been tested with local Gemma-family models.

GPU acceleration is recommended for embedding generation, document processing, and local LLM inference.

CPU-only execution is possible for some components but will be significantly slower.

## FAQ

### 1. How are financial-report tables parsed?

Documents containing structured contextual tables are first processed using **Anydoc**, which preserves the extracted content in Markdown.

### 2. What happens with scanned financial reports?

If Anydoc fails or extracts insufficient content, the pipeline falls back to **Docling**, which supports OCR and layout-aware processing.

### 3. What data is currently included in the knowledge base?

The current demonstration knowledge base primarily contains 2025 Q4 financial reports for:

* TSMC (`2330`)
* MediaTek (`2454`)

The repository also contains financial-report samples used during development and testing.

### 4. Which RAGAS version is used?

RAGAS v0.4.3 caused compatibility issues in the local evaluation environment.

After debugging the integration, the evaluation environment was downgraded to:

```text
RAGAS v0.4.1
```

### 5. Why is Qwen thinking disabled during RAGAS evaluation?

When using Qwen3.5 through LM Studio as the local judge model, thinking mode may consume a large portion of the output-token budget and repeatedly exceed the configured maximum output length.

For this reason, thinking is disabled during the RAGAS judge workflow.

### 6. Current RAGAS Evaluation Results

Comparison between `top_k = 5` and `top_k = 10`:

```text
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

Increasing `top_k` from 5 to 10 improved both Faithfulness and Context Recall in the current evaluation dataset, although Factual Correctness remained relatively low and requires further investigation.

## Known Limitations

1. The current project is a prototype and has not been designed for production-scale multi-user deployment.

2. Some underlying RAG service state is not yet fully isolated for concurrent multi-user scenarios.

3. The current deterministic calculator supports only numerical differences.

4. Evidence reuse currently uses rule-based matching of company, reporting period, and financial topic.

5. The knowledge base is intentionally small and intended for architecture demonstration rather than broad market coverage.

6. The project currently focuses on retrieving and comparing reported financial information rather than providing investment recommendations.

## License

AGPL-v3

## Key Document Processing Packages

1. [Anydoc](https://github.com/firecrawl/anydoc)
2. [Docling](https://github.com/docling-project/docling)

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

本專案為獨立開發的作品集專案，用於展示 RAG、Agentic RAG、LLM 應用與 Python 系統整合能力。

專案探索如何在地端建立台股與美股公司財報的 Retrieval-Augmented Generation（RAG）與 Agentic RAG 環境，用於財報資料檢索、跨公司比較與多輪金融問答。

[![Demo 影片](http://img.youtube.com/vi/IZwksbgmWEw/0.jpg)](https://www.youtube.com/watch?v=IZwksbgmWEw "YouTube Demo")

![多輪對話](./screenshots/multirounds-chat-terminal.png)

### 提醒

本專案仍持續開發中，目前屬於產品雛形，並非 Production-ready 系統。

## 使用前的前置作業

1. 安裝 **LM Studio**，並載入相容的本機 Instruction Model，例如 **Gemma4-E2B-Instruct**。啟用 **Developer → Server API**，並令 OpenAI-compatible API 可以透過 localhost 存取，例如：

   ```text
   http://127.0.0.1:1234
   ```

2. 若需要 GPU 加速，安裝 NVIDIA CUDA 13.1 或 12.6。

3. 安裝 Python 3.13。

4. 安裝其他 Python 套件：

   ```bash
   pip install -r requirements.txt
   ```

5. 建立財報知識庫前，修改 `financial_ingestion_advanced.py` 最後的 `pipeline.ingest_pdf(...)` 參數，設定財報檔案、股票代號、市場、年度與季度，再寫入本機 ChromaDB。

## 使用方式

啟動 Chainlit 對談介面：

```bash
chainlit run app.py -w
```

## 目前產品範圍

目前產品雛形主要聚焦於：

* 財報資料檢索
* Evidence 驗證
* 數字比較
* 跨公司財報查詢
* 多輪金融問答

目前尚未以完整的質化投資研究、估值建議或投資建議作為主要功能。

## 功能特點

### ✅ RAG 核心功能

1. **財報文件匯入**

   * 將財報內容與表格轉換為 Markdown。
   * 使用 fixed-size chunking 與 overlapping sliding window。
   * 對掃描文件或複雜版面提供 OCR fallback。

2. **Query processing 與多輪對話**

   * 將追問問題改寫或 contextualize 成可獨立理解的問題。
   * 處理公司、季度與前文主詞等指代關係。
   * 在適合的階段使用 Structured Output，讓後續程式可以可靠處理 LLM 輸出。

3. **股票實體正規化**

   * 將公司正式名稱、股票代號、俗稱、綽號與別名轉換成統一的 canonical ticker。
   * 支援台股與美股不同的股票名稱表示方式。

4. **Metadata-aware Retrieval**

   * 使用 ticker、year、quarter 等 metadata filter，限制向量搜尋範圍，降低跨公司或跨季度的錯誤檢索。

5. **Idempotent Vector Upsert**

   * 使用 deterministic ID：

   ```text
   Market-StockTicker-Year-Quarter-ChunkIndex
   ```

   重複匯入相同財報時，不會因為相同 chunk 重複產生不同的向量紀錄。

6. **知識庫與對話狀態分離**

   * ChromaDB 負責持久化保存財報知識庫。
   * Conversation state 則由應用層另外管理。

7. **Chainlit GUI**

   * 提供瀏覽器形式的聊天介面。
   * 顯示 Agentic RAG 各執行階段。
   * 支援多輪對話。
   * Agent 執行過程可透過收合式步驟查看，避免中間流程影響最終回答的閱讀體驗。

### ✅ RAGAS 評測流程

已建立在地端 RAGAS 評測流程，目前包含：

* Faithfulness
* Factual Correctness
* Context Recall

目前使用的本機裁判模型：

```text
qwen3.5-9b
```

評測資料集、中間 Retrieval 結果與評測輸出位於：

```text
RAGAS_folder/
```

### ✅ 使用 LangGraph 建立 Agentic RAG

`agentic-flow` 分支將原本的 RAG pipeline 擴充為具有狀態、條件路由與自主重試能力的 LangGraph Agentic RAG。

目前流程：

```text
使用者問題
    ↓
Contextualize Question
    ↓
Intent Router
    ↓
Research Planner
    ↓
Evidence Reuse Checker
    ↓
RAG Executor
    ↓
Evidence Checker
    ↓
┌─────────────────────────────┐
│ proceed                     │
│ retrieve again              │
│ regenerate unsupported      │
│ retry 超過限制後停止         │
└─────────────────────────────┘
    ↓
Calculator
    ↓
Report Writer
    ↓
最終回答
```

目前已實作以下 Agentic RAG 功能：

1. **Conversation Contextualization**

   * 將使用者追問改寫成可以脫離歷史對話獨立理解的問題。
   * 在適當情況保留前一輪的公司、財報期間、金融指標與比較條件。

2. **Intent Router**

   * 將使用者問題分類成：

   ```text
   single_company
   peer_comparison
   industry_research
   general_financial_question
   ```

3. **Research Planner**

   * 自動將研究問題拆解成多個獨立 Research Tasks。
   * 公司比較問題會拆成個別公司的 Retrieval Task，再由後續階段統整。

4. **RAG Executor**

   * 使用既有財報 RAG Pipeline 執行 Research Tasks。
   * 在向量搜尋前依照公司與財報期間建立 Metadata Filter。

5. **Evidence Checker**

   * 判斷取得的 Evidence 是否足以支援回答。
   * 識別 missing evidence 與 weak evidence。
   * 避免沒有被檢索資料支持的回答直接進入最終報告。

6. **Controlled Retry Loop**

   * Evidence 不足時可重新檢索。
   * Answer 無法被 Evidence 支持時，可根據既有 retrieved context 重新產生答案。
   * Retrieval 與 Regeneration 均設有最大重試次數，避免 Agent Graph 形成無限迴圈。

7. **Validated Evidence Reuse**

   * 對上一輪已經通過 Evidence Checker 的證據進行比對。
   * 若新一輪 Research Task 的公司、財報期間與 Topic 相同，可以直接重複使用 Evidence。
   * 避免適合的 follow-up question 重複執行 Vector Retrieval。

8. **Deterministic Calculator Tool**

   * 判斷使用者是否明確要求數字差額。
   * LLM 僅負責理解計算意圖與找出 operands。
   * 真正的單位正規化與數學計算由 Python 執行，避免依賴 LLM 直接做 arithmetic。
   * 目前支援：

   ```text
   difference
   ```

9. **Report Writer**

   * Evidence validation 與必要計算完成後，再產生最終回答。

10. **Chainlit Agent Execution Visualization**

    * 可以查看 LangGraph 各 Node 的執行階段與中間結果。
    * 使用收合式步驟呈現，讓最終回答仍保持容易閱讀。

## 架構亮點

### 1. 文件解析採 Strategy Pattern 與 Graceful Fallback

財報解析器透過共同的 `DocumentExtractor` 介面抽象化。

目前執行策略：

```text
Anydoc
   ↓ 解析失敗 / 取得內容不足
Docling
```

系統會先使用 [Anydoc](https://github.com/firecrawl/anydoc) 解析文件。

若解析發生 Exception，或取得的內容不足，則自動 fallback 到具備 OCR 與 Layout Processing 能力的 [Docling](https://github.com/docling-project/docling)。

因此未來需要增加或替換文件解析工具時，不需要將解析工具與 ingestion pipeline 緊密綁定。

### 2. 股票實體與 Vector DB Metadata 採 Adapter-style Normalization

系統將使用者輸入的股票名稱表示方式，與 Vector DB 實際保存的 ticker metadata 分離。

Entity Normalizer 可以處理：

* 股票代號
* 公司正式名稱
* 公司別名
* 市場常見俗稱與綽號
* 不同市場的 ticker representation

Retrieval Layer 再將 canonical entity 轉換成 ChromaDB 查詢所需要的 metadata 格式。

### 3. Query Processing 的 Graceful Degradation

非 Agentic RAG Pipeline 會優先嘗試透過 LLM 將問題轉成 Structured Query。

如果 Structured Output 失敗，例如：

* LLM 回傳 Markdown
* JSON 格式錯誤
* Structured parsing 發生 Exception

系統不會立即終止，而可以降級到較簡單的 Query Rewrite、Entity Normalization 與 Vector Retrieval 流程。

### 4. Evidence-gated Agentic Workflow

Agentic RAG 不直接相信第一次產生的回答。

取得 Evidence 後，Evidence Checker 會先判斷下一步：

```text
proceed
retrieve_again
regenerate_answer
stop
```

Retrieval 與 Answer Regeneration 均設有 Retry Limit，避免 Agent 自主迴圈無限制執行。

### 5. LLM Planning + Deterministic Execution

數值計算流程將「語言理解」與「真正的數學運算」分開：

```text
LLM
↓
理解計算意圖
找出 operands
↓
Python
↓
單位正規化
確定性數學計算
```

讓 probabilistic LLM 負責理解問題，deterministic Python 負責需要精確性的金融計算。

## 主要檔案

### Core RAG

* `myrag_module.py`

  * 財報 RAG 核心服務
  * Vector Retrieval
  * Metadata Filtering
  * Query decomposition / rewrite
  * LM Studio 整合

* `financial_ingestion_advanced.py`

  * 財報 ingestion pipeline
  * Anydoc / Docling fallback
  * Chunking
  * Embedding
  * ChromaDB upsert

* `entity_normalizer.py`

  * 公司名稱與股票代號正規化

* `rag_event.py`

  * RAG Event 結構

* `rag6_practice.py`

  * RAG 執行範例

### Agentic RAG

* `agent_state.py`

  * LangGraph 共用 State

* `agent_node.py`

  * Agent Nodes
  * Research Planning
  * Evidence Reuse
  * Evidence Validation
  * Retry Logic
  * Calculator
  * Report Writer

* `agent_graph.py`

  * LangGraph Graph
  * Conditional Routing
  * Retry Loop

* `app.py`

  * Chainlit Agentic RAG UI
  * Conversation History
  * 跨回合保留已驗證 Evidence
  * Agent 執行階段視覺化

### Evaluation

* `RAGAS_folder/ragas_generate_dataset.py`
* `RAGAS_folder/evaluate_ragas.py`
* `RAGAS_folder/base_dataset.json`

### Tests

專案亦包含針對 Agentic RAG 主要流程的測試，例如：

* Intent Router
* Agent Graph
* Answer Regeneration Loop
* Retrieval Retry Loop
* Evidence Reuse
* Calculator

## 尚未實作 / Roadmap

目前 MVP 尚未實作，或仍預計進一步擴充的功能：

1. **HyDE（Hypothetical Document Embeddings）**

2. **量化與質化研究 Router**

   * 明確區分數字檢索問題與質化商業研究問題。

3. **Hybrid Retrieval**

   * Dense Vector Retrieval
   * BM25 / Lexical Search
   * Re-ranking

4. **金融術語正規化**

   * 建立中央化 Financial Terminology Normalization。
   * 處理金融指標的正式名稱、縮寫、別名與同義詞。

5. **更多 Deterministic Financial Tools**

   * Ratio
   * Growth Rate
   * Percentage Comparison
   * 多公司數值分析

6. **更完整的質化金融研究**

   * Fundamental Analysis
   * Management Outlook
   * 公司營運與產業解讀

7. **Multi-hop Agentic RAG Evaluation**

   * 驗證需要組合多份 Evidence 才能回答的問題品質。

8. **LangGraph-native Fan-out / Fan-in**

   * 將不同 Research Tasks 直接轉成 Graph 層級的平行分支，再統一彙整結果。

9. **擴充財報 Knowledge Base**

   * 更多公司
   * 更多季度
   * 更多台股與美股資料

## 系統需求

本專案以地端執行為主要設計目標。

目前開發與驗證環境：

* NVIDIA RTX 4080 Laptop GPU
* 64 GB 系統記憶體
* Python 3.13
* LM Studio
* Local LLM
* ChromaDB

目前已使用 Gemma 系列本機模型進行測試。

Embedding、文件解析與本機 LLM Inference 建議使用 GPU 加速。

部分元件亦可使用 CPU-only 執行，但速度會明顯較慢。

## FAQ

### 1. 財報中的表格如何解析？

具有結構與上下文的財報文件會優先使用 **Anydoc** 解析，並將結果保存為 Markdown。

### 2. 掃描版財報如何處理？

如果 Anydoc 解析失敗，或抽取出的內容不足，Pipeline 會自動 fallback 到 **Docling**，使用 OCR 與 Layout-aware processing 處理。

### 3. 目前 Knowledge Base 有哪些資料？

目前 Demo Knowledge Base 主要包含：

* 台積電（TSMC，`2330`）2025 Q4 財報
* 聯發科（MediaTek，`2454`）2025 Q4 財報

Repository 內亦包含開發與測試期間使用的財報樣本。

### 4. 使用哪一個 RAGAS 版本？

RAGAS v0.4.3 在目前本機評測環境發生相容性問題。

經過除錯後，目前 Evaluation Environment 使用：

```text
RAGAS v0.4.1
```

### 5. 為什麼 RAGAS 裁判模型需要關閉 Qwen Thinking？

透過 LM Studio 使用 Qwen3.5 作為本機 Judge Model 時，Thinking Mode 容易消耗大量 Output Token Budget，導致評測過程反覆超過設定的最大 Output Token 數。

因此目前在 RAGAS Judge Workflow 中停用 Thinking。

### 6. 目前 RAGAS 評測結果

`top_k = 5` 與 `top_k = 10` 比較：

```text
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

以目前 Evaluation Dataset 而言，`top_k` 從 5 增加到 10 後，Faithfulness 與 Context Recall 均提升；Factual Correctness 則僅小幅增加，仍屬於後續需要分析與改善的項目。

## 已知限制

1. 本專案目前為 Prototype，尚未以 Production-scale 多使用者部署為設計目標。

2. 部分底層 RAG Service State 尚未針對 Concurrent Multi-user Scenario 完全隔離。

3. Deterministic Calculator 目前僅支援數值 `difference`。

4. Evidence Reuse 目前主要以 company、reporting period 與 financial topic 的 rule-based matching 判斷。

5. Knowledge Base 刻意維持較小規模，目前主要用於展示 RAG 與 Agentic RAG 架構，而非提供完整市場覆蓋。

6. 目前主要聚焦在已揭露財報資訊的 Retrieval 與 Comparison，不提供投資建議。

## 軟體授權

AGPL-v3

## 主要文件處理套件

1. [Anydoc](https://github.com/firecrawl/anydoc)
2. [Docling](https://github.com/docling-project/docling)

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
