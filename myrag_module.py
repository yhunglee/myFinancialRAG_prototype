from __future__ import annotations

import asyncio
import chromadb
from openai import AsyncOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
import json
import re
from entity_normalizer import StockEntityNormalizer
from typing import AsyncGenerator, TypeAlias
from rag_event import RAGEvent

RetrievedContexts: TypeAlias = dict[str, list[str]]
RetrievedMetadata: TypeAlias = dict[str, list[dict]]
RAGMessages: TypeAlias = list[dict[str, str]]


class FinancialRAGService:
  """台美股財報專用 RAG 檢索問答服務模組"""

  def __init__(
      self,
      db_path: str = "./chroma_db",
      collection_name: str = 'financial_reports',
      lm_studio_url: str = "http://localhost:1234/v1",
      llm_model: str = "local-model",
      embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      max_history_turns: int = 3,
  ) -> None:

    # 1. 初始化 Embedding 模型
    self.embedding_model = SentenceTransformer(embedding_model_name)

    # 2. 連線向量資料庫
    self.chroma_client = chromadb.PersistentClient(path=db_path)
    self.collection = self.chroma_client.get_or_create_collection(
      name=collection_name
    )

    # 3. 連線 LM Studio API
    self.llm_client = OpenAI(
      base_url=lm_studio_url,
      api_key="lm-studio" # 任意字詞，因為沒有開認證
      )
    # 專供 Chainlit UI 串流使用的非同步 Client
    self.async_llm_client = AsyncOpenAI(
      base_url=lm_studio_url,
      api_key="lm-studio" 
    )
    self.llm_model = llm_model

    # 4. 狀態管理(對話記憶與滑動視窗上限)
    self.chat_history: list[dict[str,str]] = []
    self.max_history_messages = max_history_turns * 2

    # 初始化股票名稱實體規範化引擎
    self.normalizer = StockEntityNormalizer()

  def clean_history(self) -> None:
    """重設對話歷史紀錄"""
    self.chat_history = []
  
  def build_compared_results(self, retrieved_contexts: dict) -> str:
    """
    組合檢索出的結果清單，變成系統提示詞
    Notice: 具備空資料識別
    """

    # 1. 初始化一個字串，先放入大標題
    context_str = "### 參考資料區塊：\n\n"
    has_any_content = False

    # ====
    # 使用 for 迴圈讀取字典裡的每一組資料
    for label, chunks in retrieved_contexts.items():
      if chunks: # 只有真正檢索到內容時，才產生標題與清單
        has_any_content = True
        context_str += f"【公司代號：{label}】\n"
        for chunk in chunks:
          context_str += f"- {chunk}\n"
        context_str += "\n"
      else:
        context_str += f"【公司代號：{label}】\n- （資料庫中無此公司相關財報段落）\n\n"

    return context_str if has_any_content else "無相關資料"

  def decompose_query(self, user_query: str) -> dict | None:
    """
    將使用者問題(結合對話歷史)拆解為結構化的檢索任務清單。
    支援單一公司查詢與跨公司(台股/美股)比較查詢。
    """
    # 1. 整理最近歷史對話，提供情境脈絡
    history_text = "無"
    if self.chat_history:
      history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in self.chat_history[-4:]]
      )

    # 2. 設計結構化 Prompt(包含 Few-shot 範例與 Schema 約束)
    system_prompt = (
      "你是一個金融數據檢索路由器。請分析使用者問題並拆解成獨立檢索任務。\n"
      "嚴格僅輸出標準 JSON，格式為：\n"
      '{"is_comparison": bool, "sub_queries": [{"ticker": "代號", "company_name": "名稱", "search_query": "搜尋詞"}]}'
    )

    prompt = f"[歷史對話]\n{history_text}\n\n[最新提問]\n{user_query}\n\n[JSON 輸出]:"

    try:
      # 3. 呼叫本機 LM Studio 模型
      response = self.llm_client.chat.completions.create(
        model=self.llm_model,
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
        ],
        temperature=0.0, # 0 是為了確保穩定輸出結構化任務
      )
      raw_content = response.choices[0].message.content.strip()
      

      # 4. 清除 Markdown 外衣(防禦性處理)
      cleaned_json_str = self._clean_json_output(raw_content)

      # 5. 反序列化為 Python 字典
      parsed_data = json.loads(cleaned_json_str)

      # 確保必要的 key 存在
      if "sub_queries" in parsed_data and len(parsed_data["sub_queries"]) > 0:
        for task in parsed_data["sub_queries"]:
          raw_ticker = task.get("ticker", "")
          raw_name = task.get("company_name", "")

          # 分別取得 Normalizer 的比對結果
          meta_from_name = self.normalizer.normalize(raw_name)
          meta_from_ticker = self.normalizer.normalize(raw_ticker)

          # 核心邏輯：優先信任「名稱 (company_name)」的比對結果，因為那是使用者最常輸入的實體
          # 若名稱找不到，才採信 LLM 輸出的 Ticker
          final_meta = meta_from_name or meta_from_ticker

          if final_meta:
            # 覆寫為標準化的代號與全名
            task["ticker"] = final_meta["canonical_ticker"]
            task["company_name"] = final_meta["formal_name"]
        return parsed_data
    except Exception as e:
      print(f"[警告] 查詢分解失敗 ({e})，準備降級至 Rewrite Query。")

    return None  # 回傳 None 代表第 1 層解析失敗

  def rewrite_query(self, user_query: str, entity_hint: str = '') -> str:
    """根據對話歷史將代名詞或簡短問題改寫為完整檢索語句 
    a.k.a. 重寫使用者的問題，為了解決指代消除問題"""

    if not self.chat_history:
      return user_query

    # 只取最近 2~3 輪的[使用者提問]與[簡短摘要]，避免干擾
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history[-4:]])

    prompt = f"""你是一位搜尋改寫助理。請參考對話歷史，將最後一個問題改寫成包含明確主詞(公司名、年份、季度、指標)的獨立查詢。
    [已解析股票實體]
    {entity_hint}

    [歷史對話]
    {history_text}
    
    [最新提問]
    {user_query}

    [改寫查詢] (直接輸出改寫後的問題，不要加任何贅字):
    """
    try:
        res = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[警告] rewrite_query 呼叫失敗: {e}")
        return user_query

  def retrieve(
      self,
      search_query: str,
      top_k: int = 5,
      where_filter: dict | None = None
  ) -> tuple[list[str], list[dict]]:
    """執行純向量搜尋(支援選擇性 Metadata Filtering)"""

    query_vector = self.embedding_model.encode(
      [search_query], normalize_embeddings=True
    ).tolist()

    query_params = {
      "query_embeddings": query_vector,
      "n_results": top_k
    }

    # 使用原始 where_filter 檢索
    if where_filter:
      query_params["where"] = where_filter

    results = self.collection.query(**query_params)
    
    documents = (
      results["documents"][0]
      if results.get("documents")
      else []
    )

    metadatas = (
      results["metadatas"][0]
      if results.get("metadatas")
      else []
    )

    return documents, metadatas


  def rag_chat(
      self,
      user_query: str,
      top_k: int = 5,
      where_filter: dict | None = None
    ) -> tuple[str, RetrievedContexts, RetrievedMetadata]:
    """(同步回應版)核心問答流程：問題改寫 -> 向量檢索 -> 增強生成 -> 更新記憶
    第二階段: 檢索與生成"""
    messages, retrieved_contexts, retrieved_metadata = self._rag_core(
      user_query,
      top_k,
      where_filter
      )

    # 生成回答
    response = self.llm_client.chat.completions.create(
      model=self.llm_model,
      messages=messages,
      temperature=0.1, # 財報重視精確，維持低溫度
      )
    answer = response.choices[0].message.content.strip()

    self._update_message_history(user_query, answer)

    return answer, retrieved_contexts, retrieved_metadata

  def _update_message_history(self, user_query, answer):
    """更新記憶體裡的對話歷史"""
    
    # 更新記憶體對話歷史(純 Python 操作，不碰 ChromaDB)
    self.chat_history.append({"role": "user", "content": user_query})
    self.chat_history.append({"role": "assistant", "content": answer})

    # 保持記憶長度，避免 token 受限(最近六則訊息)
    if len(self.chat_history) > self.max_history_messages:
      self.chat_history = self.chat_history[-self.max_history_messages:]

  def _rag_core(
      self,
      user_query: str,
      top_k: int = 5,
      where_filter: dict | None = None
    ) -> tuple[RAGMessages, RetrievedContexts, RetrievedMetadata]:
    """ RAG 核心流程的共用函式: 問題改寫 -> 向量檢索 -> 增強生成 """
    print(f"\n[系統] 收到使用者提問: {user_query}")
    retrieved_contexts = {}
    retrieved_metadata = {}

    # 透過 Normalizer 自動從句中抓取所有提及的廠商
    extracted_entities = self.normalizer.extract_entities_from_text(user_query)

    entity_hint = ""
    if extracted_entities:
      entity_hint = "\n".join(
        (
          f"{entity['matched_alias']} = " 
          if entity.get("matched_alias") 
          else ""
        )
        + f"{entity['formal_name']}"
        f"({entity['canonical_ticker']})"
        for entity in extracted_entities
      )

    """
    階段一：指代消解 (Coreference Resolution)
    不論如何，先讓 LLM 根據歷史對話把代名詞補齊
    """
    refined_query = self.rewrite_query(user_query, entity_hint=entity_hint)
    print(f"[管線狀態] 改寫後的搜尋語句: {refined_query}")

    """
    階段二: 意圖與實體擷取(Intent & Entity Extraction)
    用改寫後的、語意完整的句子去進行 JSON 拆解
    """
    plan = self.decompose_query(refined_query)
    if plan and len(plan.get("sub_queries", [])) > 0:
      print("[管線狀態] 成功命中【第 1 層：結構化分解】")

      for task in plan.get("sub_queries", []):
        raw_ticker = task.get("ticker")
        sub_query = task.get("search_query", refined_query)
      
      # 進入資料庫檢索前，進行格式搭配
        db_ticker = self._adapt_ticker_for_db(raw_ticker)
        current_where = {"ticker": db_ticker} if db_ticker else None

        docs, metadatas = self.retrieve(search_query=sub_query, top_k=top_k, where_filter=current_where)

        label = raw_ticker if raw_ticker else task.get("company_name", "通用資料")

        retrieved_contexts[label] = docs
        retrieved_metadata[label] = metadatas
    else:
    # 2. 降級到第 2 層，純文字改寫(Rewrite Query)
      print("[管線狀態] 觸發降級機制 -> 執行【第 2 層：純文字語意改寫】")

      if extracted_entities:
      # 即使 LLM 沒輸出 JSON，只要句子中有提到公司，分別執行 Metadata 檢索
        for entity in extracted_entities:
          ticker = entity["canonical_ticker"]
          db_ticker = self._adapt_ticker_for_db(ticker)
          where_filter = {"ticker": db_ticker}

          docs, metadatas = self.retrieve(search_query=refined_query,
                              top_k=top_k,
                              where_filter=where_filter)
          
          label = f"{entity['formal_name']} ({ticker})"

          retrieved_contexts[label] = docs
          retrieved_metadata[label] = metadatas
      else:
      # 若完全找不到對應股票，才退為全域搜尋
        docs, metadatas = self.retrieve(search_query=refined_query,
                            top_k=top_k*2, where_filter=None)
        retrieved_contexts["全域搜尋"] = docs
        retrieved_metadata["全域搜尋"] = metadatas

    # 3. 組裝 Context 與送出生成
    context_text = self.build_compared_results(retrieved_contexts)

    # 組裝發送給 LLM 的完整 Prompt
    system_prompt = (
      "你是一名專業的台美股財報分析助理。請嚴格依據提供的參考資料與對話歷史回答。"
      "若進行跨公司比較，請務必使用 Markdown 表格對齊數據，並標註資料來源。"
      "若資料未提及，請直接回答[財報未揭露]，嚴禁自行捏造數據。"
    )
    user_content = (f"""
      [實體解析]
      {entity_hint or "無"}

      [改寫後問題]
      {refined_query}

      [參考資料]
      {context_text}
      
      [使用者原始問題]
      {user_query}
      """
    )

    messages = [{"role": "system", "content": system_prompt}]

    # 帶入最近兩輪歷史對話，維持語境連貫
    messages.extend(self.chat_history[-self.max_history_messages:])
    messages.append({"role": "user", "content": user_content})
    return messages, retrieved_contexts, retrieved_metadata

  async def rag_chat_stream(self, user_query: str, top_k: int = 5 , where_filter: dict | None = None) -> AsyncGenerator[RAGEvent, None] :
    """(串流回應版)核心問答流程：問題改寫 -> 向量檢索 -> 增強生成 -> 更新記憶
        第二階段: 檢索與生成"""
    # 利用 to_thread 將耗時的同步前處理丟到背景執行，釋放 Event Loop
    messages, retrieved_contexts, retrieved_metadata = await asyncio.to_thread(
        self._rag_core, user_query, top_k, where_filter
    )
    response = await self.async_llm_client.chat.completions.create(
      model=self.llm_model,
      messages=messages,
      temperature=0.1,
      stream=True, # 啟用串流輸出
    )

    full_answer = ''

    async for chunk in response:
      # 避免 chunk.choices 為空的極端 API 邊界狀況
      if not chunk.choices:
          continue
          
      delta = chunk.choices[0].delta
      
      # 攔截原生支援的推理欄位 (reasoning_content)
      reasoning_token = getattr(delta, "reasoning_content", None)
      if reasoning_token:
        yield RAGEvent(
          type="reasoning",
          content=reasoning_token,
        )

      # 攔截回答內容
      content_token = getattr(
        delta,
        "content",
        None
      )
      if content_token:
        full_answer += content_token
        yield RAGEvent(
          type="answer",
          content=content_token
        )

    # 更新對話歷史
    self._update_message_history(user_query, full_answer)

  def _clean_json_output(self, raw_text: str) -> str:
    """利用正規表示式，取得字串中最外層的 JSON 物件"""
    # 移除前後的 ```json 與 ```
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
      cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
      cleaned = cleaned[3:]

    if cleaned.endswith("```"):
      cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # 若前後仍有雜訊，以正規擷取 { ... } 區塊
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else cleaned

  def _adapt_ticker_for_db(self, canonical_ticker: str | None) -> str | None:
    """
    資料庫 adaptor：將正規化代號 (e.g., 2330.TW, NVDA) 
    轉換為目前 ChromaDB 實際儲存的格式(e.g.: 2330)。
    保留未來切換不同資料庫或擴充欄位的彈性。
    """
    if not canonical_ticker:
        return None
        
    # 針對台股，過濾掉 .TW 後綴以符合現有 Chunk 的 Metadata
    if canonical_ticker.endswith(".TW"):
        return canonical_ticker.replace(".TW", "")
        
    # 美股或其他不帶後綴的代號直接回傳
    return canonical_ticker