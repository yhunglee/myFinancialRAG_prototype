from __future__ import annotations
from typing import Literal, TypedDict

class FinancialResearchState(TypedDict):
  # 使用者這一輪輸入的原始問題
  question: str

  """
  將對話上下文補齊後，
  可以獨立理解的完整問題
  """
  standalone_question: str

  # 目前 Chainlit session 的歷史對話
  chat_history: list[dict]


  # intent_router 輸出
  intent: str
  companies: list[str]
  periods: list[str]
  router_confidence: float

  # research_planner 輸出
  research_plan: list

  # 存放是否要重複使用先前的 evidence
  previous_validated_evidences: list
  reuse_evidence: bool
  
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

  # retrieval_again 使用
  retrieval_count: int

  # report_writer 輸出
  final_answer: str
