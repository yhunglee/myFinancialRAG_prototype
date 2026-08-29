from __future__ import annotations
from typing import TypedDict

class FinancialResearchState(TypedDict):
  # 使用者原始問題
  question: str

  # intent_router 輸出
  intent: str
  companies: list[str]
  periods: list[str]
  router_confidence: float

  # research_planner 輸出
  research_plan: list
  
  # rag_executer 使用
  current_task: int
  evidence: list

  # Evidence_checker 使用
  sufficient: bool
  missing_information: list[str]

  # report_writer 輸出
  final_answer: str
