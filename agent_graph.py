from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import (
  contextualize_question,
  intent_router,
  research_planner,
  evidence_reuse_checker,
  rag_executor,
  evidence_checker,
  answer_regenerator,
  retrieve_again,
  calculator,
  report_writer,
  failure_report_writer,
)

from fanout_nodes import (
  dispatch_research_tasks,
  research_task_worker,
  fan_in_gate,
  route_after_fan_in,
)


MAX_REGENERATION_ATTEMPTS = 2
MAX_RETRIEVAL_ATTEMPTS = 2


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

  if next_action == 'retrieve_again':
    retrieval_count = state.get(
      "retrieval_count",
      0,
    )

    if retrieval_count >= MAX_RETRIEVAL_ATTEMPTS:
      return "stop"

  return next_action

def route_after_global_evidence_check(
  state: FinancialResearchState,
) -> str:
  """
  全域 Evidence Checker 完成後的路由。

  proceed:
    證據驗證通過。

  regenerate_answer:
    財務數值與證據不一致，
    使用現有 Evidence 重新產生答案。

  stop:
    無法修正，或達到重試上限。
  """

  next_action = state.get(
    "next_action",
    "stop",
  )

  if (
    state.get("sufficient", False)
    and next_action == "proceed"
  ):
    return "proceed"

  if next_action == "regenerate_answer":
    regeneration_count = state.get(
      "regeneration_count",
      0,
    )

    if regeneration_count >= MAX_REGENERATION_ATTEMPTS:
      return "stop"

    return "regeneration_answer"
  
  return "stop"

def build_agent_graph():
  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node("contextualize_question", contextualize_question)
  graph.add_node("intent_router", intent_router)
  graph.add_node("research_planner", research_planner)
  graph.add_node("evidence_reuse_checker", evidence_reuse_checker)
  graph.add_node("research_task_worker", research_task_worker)
  graph.add_node("fan_in_gate", fan_in_gate)

  graph.add_node("evidence_checker", evidence_checker)
  graph.add_node("answer_regenerator", answer_regenerator)
  graph.add_node("retrieve_again", retrieve_again)
  graph.add_node("calculator", calculator)
  graph.add_node("report_writer", report_writer)
  graph.add_node("failure_report_writer", failure_report_writer)

  # 定義執行流程
  graph.add_edge(START, "contextualize_question")
  graph.add_edge("contextualize_question", "intent_router")
  graph.add_edge("intent_router", "research_planner")
  graph.add_edge("research_planner", "evidence_reuse_checker")
  graph.add_edge("research_task_worker", "fan_in_gate")
  graph.add_conditional_edges(
    "fan_in_gate",
    route_after_fan_in,
    {
      "proceed": "evidence_checker",
      "stop": "failure_report_writer"
    }
  )

  # Evidence conditional routing
  graph.add_conditional_edges(
    "evidence_checker",
    route_after_global_evidence_check,
    {
      "proceed": "calculator",
      "regenerate_answer": "answer_regenerator",

      # regeneration 超過次數限制
      "stop": "failure_report_writer"
    }
  )

  graph.add_conditional_edges(
    "evidence_reuse_checker",
    dispatch_research_tasks,
  )

  """
  Answer 修正完成後，
  必須重新接受 Evidence checker 驗證
  """
  graph.add_edge("answer_regenerator", "evidence_checker")

  graph.add_edge("calculator", "report_writer")

  graph.add_edge("report_writer", END)

  graph.add_edge("failure_report_writer", END)

  return graph.compile()