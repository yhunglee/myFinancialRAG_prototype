from __future__ import annotations
from typing import Literal, TypedDict

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
  weak_evidence: list[str]
  unsupported_answer: list[str]

  failure_type: Literal[
    "none",
    "missing_evidence",
    "weak_evidence",
    "answer_not_supported"
  ]

  next_action: Literal[
    "proceed",
    "retrieve_again",
    "regenerate_answer"
  ]

  # answer_regenerator 使用
  regeneration_count: int

  # report_writer 輸出
  final_answer: str
