from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import (
  intent_router,
  research_planner,
  rag_executor,
  evidence_checker,
)

def route_after_evidence_check(
    state: FinancialResearchState,
) -> str:
  """
  根據 evidence_checker 的結果，
  決定下一個執行方向。
  """

  return state["next_action"]

def build_agent_graph():
  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node("intent_router", intent_router)
  graph.add_node("research_planner", research_planner)
  graph.add_node("rag_executor", rag_executor)
  graph.add_node("evidence_checker", evidence_checker)

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
      "retrieve_again": END,
      "regenerate_answer": END,
    }
  )

  return graph.compile()