from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


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
