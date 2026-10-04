from __future__ import annotations
import operator
from typing import (
  Annotated, 
  Literal, 
  TypedDict
)

class ResearchTaskState(TypedDict):
  """
  單一 ResearchTask 在 fan-out branch 中使用的 State。

  每個 task 都有自己的 retrieval_count，
  因此其中一個 task retry 時，
  不會影響其他已成功的 task。
  """

  # research_palnner 產生的單一 task
  task: dict

  # 這個 task 取得的 evidence
  task_evidence: dict | None

  # task-level evidence check
  sufficient: bool

  failure_type: Literal[
    "none",
    "missing_evidence",
    "weak_evidence",
  ]

  next_action: Literal[
    "proceed",
    "retrieve_again",
    "stop",
  ]

  # 只屬於這個 task 的 retry counter
  retrieval_count: int

  """
  task 完成後，將 evidence 回傳給 parent graph
  
  使用 list 是為了之後可以透過 reduce 
  合併不同的 parallel branches 的結果。
  """
  evidence: list[dict]


  

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
  previous_validated_evidence: list
  reuse_evidence: bool
  
  # rag_executer 使用
  current_task: int

  """
  fan-out / fan-in 使用。
  
  每個 parallel ResearchTask branch 
  都可以回傳:
  {"evidence": [task_evidence]}
  
  operator.add reducer 會自動合併成:
  [
    task_1_evidence,
    task_2_evidence,
    ...
  ]
  """
  evidence: Annotated[
    list[dict],
    operator.add
  ]


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

  """
  舊版 global retrieval retry counter。
  暫時保留，因為舊的 retrieval_again() 仍使用。
  """
  # retrieval_again 使用
  retrieval_count: int

  # calculator 使用
  calculation_required: bool
  calculation_results: list[dict]

  # report_writer 輸出
  final_answer: str

