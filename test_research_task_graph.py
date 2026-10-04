from __future__ import annotations

import asyncio

from research_task_graph import (
  bulid_research_task_again,
)

async def main():

  graph = bulid_research_task_again()

  initial_state = {
    "task": {
      "task_id": "task_1",
      "company": "台積電",
      "period": "2025Q4",
      "topic": "營收",
      "query": "台積電 2025 年第四季營收",
    },

    "task_evidence": None,

    "sufficient": False,

    "failure_type": "none",

    "next_action": "proceed",

    "retrieval_count": 0,

    "evidence": []
  }

  result = await graph.ainvoke(
    initial_state
  )

  print(
  "\n=========================="
  )
  print(
    "Validation"
  )
  print(
    "=========================="
  )

  print(
    "sufficient:",
    result["sufficient"]
  )

  print(
    "retrieval_count:",
    result["retrieval_count"]
  )

  print(
    "evidence count:",
    len(result["evidence"])
  )


  assert result["sufficient"] is True

  assert result["retrieval_count"] == 0
  assert result["task_evidence"] is not None

  assert len(result["evidence"]) == 1

  print(
    "\n[PASS] Research task graph "
    "completed without retry."
  )


if __name__ == '__main__':
  asyncio.run(main())