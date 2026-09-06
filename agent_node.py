from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

from agent_state import FinancialResearchState
from myrag_module import FinancialRAGService

client = OpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio"
)

MODEL_NAME = 'local_model'

rag_service = FinancialRAGService(
  db_path="./chroma_db",
  collection_name="financial_reports",
)

class RouterResult(BaseModel):
  """
  自然語言問題, 轉成程式可以理解的決策
  """
  intent: Literal[
    "single_company",
    "peer_comparison",
    "industry_research",
    "general_financial_question"
  ]

  companies: list[str]
  periods: list[str]
  confidence: float

def intent_router(state: FinancialResearchState) -> dict:
  """
  判斷使用者問題屬於哪一種金融研究意圖。
  
  Node Input:
    FinancialResearchState
    
  Node output:
    更新 intent / companies / periods / router_confidence
  """

  question = state["question"]

  

  system_prompt = """
  你是一個金融研究問題路由器。
  請判斷使用者問題屬於以下哪一種 intent:
  
  1. single_company
  單一公司財報、營收、獲利、展望等問題。
  
  2. peer_comparison
  比較兩家或多家公司。
  
  3. industry_research
  產業研究、上下游研究、供應鏈研究、產業公司群研究
  
  4. general_financial_question
  不針對特定公司的通用金融問題。
  
  companies:
  找出問題中涉及的公司股票代號。
  如果無法判斷股票代號，可以先保留使用者使用的公司名稱
  
  periods:
  找出問題中的年度或季度，例如:
  2025Q4
  2025
  2026Q1
  
  如果沒有期間則回傳空陣列。
  
  confidence:
  0 到 1 之間，表示你對 intent 判斷的信心。
  """
  response = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": question
      }
    ],
    response_format=RouterResult,
    temperature=0,
  )

  router_result = response.choices[0].message.parsed

  if router_result is None:
    raise ValueError("intent_router 無法取得 structured output")

  return {
    "intent": router_result.intent,
    "companies": router_result.companies,
    "periods": router_result.periods,
    "router_confidence": router_result.confidence,
  }

class ResearchTask(BaseModel):
  """
  小問題(工作項目)
  """
  company: str | None
  period: str | None
  topic: str
  query: str

class ResearchPlan(BaseModel):
  """
  複雜問題,拆成多個可以交給 RAG 執行的小問題
  """
  research_goal: str
  tasks: list[ResearchTask]

class Evidence(BaseModel):
  """
  單一 ResearchTask 執行後取得的研究證據。
  """
  task_id: str
  company: str | None
  period: str | None
  query: str
  answer: str

  retrieved_contexts: list[str] # 實際送進 LLM 的文字內容，後續做 RAGAS、faithfulness 或 evidence checking 很有用

  """
  向量資料庫檢索結果的原始 metadata，例如 ticker、year、quarter 等,
  backend / debug / evaluation 用
  """
  metadata: list[dict] 

  """
  # 從 metadata 整理後要給 UI 或最終回答顯示的引用來源，例如檔名、公司、季度、文件標題
  Chainlit / report_writer / citation UI 用
  """
  sources: list[dict] 

class EvidenceCheckResult(BaseModel):
  """
  Evidence 檢查結果
  """
  sufficient: bool
  missing_topics: list[str]
  weak_evidence: list[str]
  retry_required: bool


def research_planner(state):
  """
  根據 intent_router 的結果，
  將研究問題拆成可以交給 RAG executor 執行的 ResearchTask。
  """

  question = state["question"]
  intent = state["intent"]
  companies = state["companies"]
  periods = state["periods"]

  system_prompt = """
  You are a research planner for a financial-report RAG system.

  Your job is to convert a user's financial research question
  into small, independent retrieval tasks that can be
  executed by a RAG system.
  
  The "topic" field must be a short financial topic or metric,
  not an instruction or sentence.
  Examples:
  "營收"
  "毛利率"
  "營業利益"
  "EPS"
  "資本支出"

  Rules:

  1. Each task must represent ONE retrieval objective.
  2. Each task should normally contain only ONE company.
  3. Each task should normally contain only ONE reporting period.
  4. The query must be standalone and understandable without conversation history.
  5. Only create tasks that can be answered from company financial reports.
  6. Do not answer the user's question.
  7. Do not invent companies, reporting periods, or financial facts.
  8. Use only companies and periods supplied by the router when they are available.
  9. For company comparisons, retrieve evidence for each company separately.
  10. Do not create a separate comparison task. Comparison will be performed later by the report writer.
  11. Keep the plan as small as possible while still answering the research goal.
  """

  user_prompt = f"""
  Original question:
  {question}

  Router result:

  intent:
  {intent}

  companies:
  {companies}

  periods:
  {periods}

  Create the research plan.
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      }
    ],
    response_format=ResearchPlan,
    temperature=0,
  )

  plan = completion.choices[0].message.parsed

  if plan is None:
    raise ValueError("research_planner failed to generate ResearchPlan")

  research_plan = []

  for index, task in enumerate(plan.tasks, start=1):

    """
    Notice: Structured Output 階段，暫時驗證 LLM 輸出
    等進到 LangGraph State 後再改成 Python dict
    """
    task_dict = task.model_dump()
    task_dict["task_id"] = f"task_{index}"
    research_plan.append(task_dict)

  return {
    "research_plan": research_plan,
    "current_task": 0,
  }

def build_sources(metadata_list: list[dict]) -> list[dict]:
  return [
    {
      "ticker": metadata.get("ticker"),
      "market": metadata.get("market"),
      "year": metadata.get("year"),
      "quarter": metadata.get("quarter"),
      "chunk_index": metadata.get("chunk_index"),
    }
    for metadata in metadata_list
  ]

def rag_executor(state: FinancialResearchState):
  """
  依序執行 research_planner 產生的 ResearchTask，
  並將每個任務的 RAG 結果整理成 Evidence。
  """

  evidence: list[dict] = []

  for task in state["research_plan"]:

    answer, retrieved_contexts, retrieved_metadata = rag_service.rag_task(
      user_query=task["query"],
      top_k=5,
    )

    # _rag_core() 回傳 dict[str, list[str]]
    # Evidence 使用 list[str]，所以要攤平成單一 list
    context_list = [
      context
      for contexts in retrieved_contexts.values()
      for context in contexts
    ]

    # _rag_core() 回傳 dict[str, list[str]]
    # Evidence 使用 list[str]，所以要攤平成單一 list
    metadata_list = [
      metadata
      for metadatas in retrieved_metadata.values()
      for metadata in metadatas
    ]

    # Notice: MVP 階段直接從 metadata 建立 sources
    # 後續 Chainlit UI 可以再做專門的 source formatter
    sources = build_sources(metadata_list)

    task_evidence = Evidence(
      task_id=task["task_id"],
      company=task.get("company"),
      period=task.get("period"),
      query=task.get("query"),
      answer=answer,
      retrieved_contexts=context_list,
      metadata=metadata_list,
      sources=sources
    )

    evidence.append(
      task_evidence.model_dump()
    )

  return {
    "evidence": evidence,
    "current_task": len(state["research_plan"])
  }

def evidence_checker(state: FinancialResearchState) -> dict:
  """
  檢查 rag_executor 取得的 evidence，
  判斷目前資料是否足以完成原始研究問題。

  Node Input:
    question
    research_plan
    evidence

  Node Output:
    sufficient
    missing_information
    weak_evidence
    retry_required

  """

  question = state["question"]
  research_plan = state["research_plan"]
  evidence = state["evidence"]

  # 沒有任何 evidence 時，不需要浪費一次 LLM call
  if not evidence:
    return {
      "sufficient": False,
      "missing_information": ["No evidence was retrieved."],
      "weak_evidence": [],
      "retry_required": True,
    }

  system_prompt = """
  You are an evidence checker for a financial-report RAG system.

  Your job is NOT to answer the user's financial question.

  Your job is to determine whether the retrieved evidence is sufficient
  to answer the original research question.

  Evaluate ONLY the supplied research plan and evidence.
  Do not use outside knowledge.
  Do not invent financial facts.

  Rules:

  1. Check whether every necessary research task has supporting evidence.

  2. Evidence is considered useful only when the retrieved
     contexts contain information relevant to that task.

  3. An answer alone is not enough.
     The answer should be supported by its retrieved context.
  
  4. For company comparisons, evidence for every required company
     must be available.

  5. For multi-period questions, evidence for every required period
     must be available.

  6. If important information is missing, set sufficient to false.

  7. missing_topics should describe what information is still required.

  8. weak_evidence should identify evidence that exists but is imcomplete,
  ambiguous, irrelevant, or poorly supported by retrieved contexts.

  9. retry_required should be true when additional retrieval is needed.

  10. Do not perform the final comparison or write the final report.
  """

  user_prompt = f"""
  Original question:
  {question}

  Research plan:
  {research_plan}

  Retrieved evidence:
  {evidence}

  Determine whether the evidence is sufficient to answer the original question.
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      }
    ],
    response_format=EvidenceCheckResult,
    temperature=0,
  )

  check_result = completion.choices[0].message.parsed

  if check_result is None:
    raise ValueError(
      "evidence_checker failed to generate EvidenceCheckResult"
    )

  return {
    "sufficient": check_result.sufficient,
    "missing_information": check_result.missing_topics,
    "weak_evidence": check_result.weak_evidence,
    "retry_required": check_result.retry_required,
  }