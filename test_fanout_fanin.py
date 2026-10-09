from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import patch

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from fanout_nodes import (
  dispatch_research_tasks,
  research_task_worker,
  fan_in_gate,
  route_after_fan_in,
)

async def main():

  calls = Counter()

  class FakeSubgraph:

    async def ainvoke(self, state):

      task = state["task"]
      task_id = task["task_id"]

      # 模擬每個 task 的執行次數
      attempts = (
        1 if task_id == "task_1" else 2
      )

      calls[task_id] += attempts

      print(
        f"[TEST] {task_id}: "
        f"{attempts} retrieval attempts"
      )

      return {
        **state,
        "sufficient": True,
        "retrieval_count": attempts - 1,
        "failure_type": "none",
        "evidence": [
          {
            "task_id": task_id,
            "answer": f"Validated answer for {task_id}",
          }
        ]
      }

  def entry(state):
    return {}

  with patch(
    "fanout_nodes.build_research_task_graph",
    return_value=FakeSubgraph(),
  ):

    # 建立最小 Parent Graph，專門測 fan-out /fan-in 
    graph = StateGraph(FinancialResearchState)

    graph.add_node("entry", entry)
    graph.add_node(
      "research_task_worker",
      research_task_worker,
    )

    graph.add_node("fan_in_gate", fan_in_gate)

    graph.add_edge(START, "entry")

    graph.add_conditional_edges(
      "entry",
      dispatch_research_tasks,
    )

    graph.add_edge(
      "research_task_worker",
      "fan_in_gate",
    )

    graph.add_conditional_edges(
      "fan_in_gate",
      route_after_fan_in,
      {
        "proceed": END,
        "stop": END,
      }
    )

    app = graph.compile()

    result = await app.ainvoke({
      "reuse_evidence": False,
      "research_plan": [
        {
          "task_id": "task_1",
          "company": "台積電",
        },
        {
          "task_id": "task_2",
          "company": "聯發科"
        }
      ],
      "evidence": [],
      "task_results": [],
      "sufficient": False,
      "failure_type": "none",
    })

  print("\nTask calls: ", dict(calls))
  print("Task results:", result["task_results"])
  print("Evidence count:", len(result["evidence"]))

  assert calls["task_1"] == 1
  assert calls["task_2"] == 2

  assert result["sufficient"] is True
  assert len(result["task_results"]) == 2
  assert len(result["evidence"]) == 2

  assert {
    item["task_id"] for item in result["evidence"]
  } == {"task_1", "task_2"}

  print(
    "\n[PASS] Fan-out / Fan-in aggregation test."
  )

if __name__ == '__main__':
  asyncio.run(main())
