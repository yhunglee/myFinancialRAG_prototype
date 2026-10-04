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
    "\n====================="
  )

  print(
    "Research Task Graph Result"
  )

  print("=====================")

  print(
    "task: ",
    result['task'],
  )

  print(
    "\ntask_evidence:"
  )

  print(result['task_evidence'])

  print("\nevidence:")
  print(result['evidence'])


if __name__ == '__main__':
  asyncio.run(main())