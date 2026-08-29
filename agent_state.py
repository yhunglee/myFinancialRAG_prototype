from __future__ import annotations
from typing import TypedDict

class FinancialResearchState(TypedDict):

  question: str
  research_type: str
  companies: list[str]
  period: str
  research_plan: list
  current_task: int
  evidence: list
  sufficent: bool
  missing_information: list[str]
  final_answer: str
  