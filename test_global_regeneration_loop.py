from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_graph import route_after_global_evidence_check
from agent_node import (
  answer_regenerator,
  check_financial_unit_consistency,
)

def main():

  checker_calls = 0

  regeneration_calls = 0

  # 模擬全域 Evidence checker
  def fake_evidence_checker(state):
    nonlocal checker_calls
    checker_calls += 1

    issues = []

    for item in state["evidence"]:
      issues.extend(
        check_financial_unit_consistency(
          answer=item["answer"],
          retrieved_contexts=item["retrieved_contexts"]
        )
      )

    if issues:
      return {
        "sufficient": False,
        "failure_type": "answer_not_supported",
        "next_action": "regenerate_answer",
        "unsupported_answer": issues,
      }

    return {
      "sufficient": True,
      "failure_type": "none",
      "next_action": "proceed",
      "unsupported_answer": []
    }


  # 包裝正式的 Answer Regenerator ，記錄執行次數
  def tracked_regenerator(state):
    nonlocal regeneration_calls
    regeneration_calls += 1
    return answer_regenerator(state)

  def fake_calculator(state):
    return {"calculation_results": []}

  def fake_failure_writer(state):
    return {"final_answer": "Validation failed"}

  # 模擬 LM Studio 的 structured output
  fake_completion = SimpleNamespace(
    choices=[
      SimpleNamespace(
        message=SimpleNamespace(
          parsed=SimpleNamespace(
            answer="1,046.09 billion"
          )
        )
      )
    ]
  )

  # 建立最小 Graph，測試實際 regeneration loop
  graph = StateGraph(FinancialResearchState)
  graph.add_node(
    "evidence_checker", fake_evidence_checker,
  )
  graph.add_node(
    "answer_regenerator", tracked_regenerator
  )
  graph.add_node("calculator", fake_calculator)
  graph.add_node("failure_report_writer", fake_failure_writer)
  graph.add_edge(START, "evidence_checker")

  graph.add_conditional_edges(
    "evidence_checker",
    route_after_global_evidence_check,
    {
      "proceed": "calculator",
      "regenerate_answer": "answer_regenerator",
      "stop": "failure_report_writer"
    }
  )

  graph.add_edge(
    "answer_regenerator", "evidence_checker"
  )

  graph.add_edge("calculator", END)

  graph.add_edge("failure_report_writer", END)

  app = graph.compile()

  # task_1 故意提供錯誤答案
  # task_2 提供一致的答案，不應被修改
  initial_state = {
    "evidence": [
      {
        "task_id": "task_1",
        "query": "台積電 2025Q4 營收",
        "answer": "100 億",
        "retrieved_contexts": [
          "NT$ billions. Net Revenue: 1,046.09"
        ],
      },
      {
        "task_id": "task_2",
        "query": "聯發科 2025Q4 營收",
        "answer": "150 million",
        "retrieved_contexts": [
          "150 million"
        ],
      }
    ],
    "regeneration_count": 0,
    "sufficient": False,
    "failure_type": "none",
    "next_action": "proceed",
    "unsupported_answer": []
  }

  with patch(
    "agent_node.client.chat.completions.parse",
    return_value=fake_completion,
  ) as mock_llm:
    result = app.invoke(initial_state)

    # 只應該呼叫 llm 修正一次
    assert mock_llm.call_count == 1


  evidence_map = {
    item["task_id"]: item
    for item in result["evidence"]
  }

  print("\n===== Global Regeneration Test ======")
  print("Regeneration calls:", regeneration_calls)
  print("Regeneration count:", result["regeneration_count"])
  print("Evidence count:", len(result["evidence"]))
  print("Final sufficient:", result["sufficient"])

  # 驗證循環
  assert checker_calls == 2
  assert regeneration_calls == 1
  assert result["regeneration_count"] == 1

  # 驗證最後通過
  assert result["sufficient"] is True
  assert result["next_action"] == "proceed"

  # 驗證 reducer 沒有重複累積
  assert len(result["evidence"]) == 2

  # 台積電答案被修正
  assert (
    evidence_map["task_1"]["answer"]
    == "1,046.09 billion"
  )

  # 聯發科答案不變
  assert (
    evidence_map["task_2"]["answer"]
    == "150 million"
  )

  assert "final_answer" not in result
  print("\n[PASS] Global regeneration loop executed successfully.")

if __name__ == '__main__':
  main()