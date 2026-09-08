from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import (
  intent_router,
  research_planner,
  rag_executor,
  evidence_checker,
  answer_regenerator,
)


MAX_REGENERATION_ATTEMPTS = 2
MAX_RETRIEVAL_ATTEMPS = 2


def route_after_evidence_check(
    state: FinancialResearchState,
) -> str:
  """
  根據 evidence_checker 決定的 next_action，
  選擇下一個執行方向。

  answer regeneration 設定最大執行次數，
  防止 Graph 無限循環。
  """

  next_action = state["next_action"]

  if next_action == 'regenerate_answer':
    regeneration_count = state.get(
      "regeneration_count",
      0,
    )

    if regeneration_count >= MAX_REGENERATION_ATTEMPTS:
      return "stop"

  return next_action

def build_agent_graph():
  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node("intent_router", intent_router)
  graph.add_node("research_planner", research_planner)
  graph.add_node("rag_executor", rag_executor)
  graph.add_node("evidence_checker", evidence_checker)
  graph.add_node("answer_regenerator", answer_regenerator)

  # 定義執行流程
  graph.add_edge(START, "intent_router")
  graph.add_edge("intent_router", "research_planner")
  graph.add_edge("research_planner", "rag_executor")
  graph.add_edge("rag_executor", "evidence_checker")

  # Evidence conditional routing
  graph.add_conditional_edges(
    "evidence_checker",
    route_after_evidence_check,
    {
      "proceed": END,

      """
      Notice: Retrieval retry 尚未實作，
      MVP 階段先停止
      """
      "retrieve_again": END,

      "regenerate_answer": "answer_regenerator",

      # regeneration 超過次數限制
      "stop": END
    }
  )

  """
  Answer 修正完成後，
  必須重新接受 Evidence checker 驗證
  """
  graph.add_edge("answer_regenerator", "evidence_checker")

  return graph.compile()