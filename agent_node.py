from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from agent_state import FinancialResearchState
from openai import OpenAI


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

  client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
  )

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
    model="local-model",
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
  task_id: str
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
  query: str
  answer: str
  retrieved_contexts: list[str]
  sources: list[dict]

class EvidenceCheckResult(BaseModel):
  """
  Evidence 檢查結果
  """
  sufficient: bool
  missing_topics: list[str]
  weak_evidence: list[str]
  retry_required: bool
