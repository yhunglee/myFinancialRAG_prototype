from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import (
  intent_router,
  research_planner,
  rag_executor,
)

def build_agent_graph():
  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node("intent_router", intent_router)
  graph.add_node("research_planner", research_planner)
  graph.add_node("rag_executor", rag_executor)

  # 定義執行流程
  graph.add_edge(START, "intent_router")
  graph.add_edge("intent_router", "research_planner")
  graph.add_edge("research_planner", "rag_executor")
  graph.add_edge("rag_executor", END)

  return graph.compile()