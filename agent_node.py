from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from agent_state import FinancialResearchState
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio"
)

MODEL_NAME = 'local_model'

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
  小問題和檢索資訊
  """
  task_id: str
  company: str
  period: str
  query: str
  answer: str

  retrieved_contexts: list[str] # 實際送進 LLM 的文字內容，後續做 RAGAS、faithfulness 或 evidence checking 很有用
  metadata: list[dict] # 向量資料庫檢索結果的原始 metadata，例如 ticker、year、quarter、page、source_file 等
  sources: list[dict] # 整理後要給 UI 或最終回答顯示的引用來源，例如檔名、頁碼、公司、季度、文件標題

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