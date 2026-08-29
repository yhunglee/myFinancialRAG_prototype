from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import intent_router

def build_agent_graph():
  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node("intent_router", intent_router)

  # 定義執行流程
  graph.add_edge(START, "intent_router")
  graph.add_edge("intent_router", END)

  return graph.compile()