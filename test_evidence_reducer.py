from agent_state import merge_evidence_by_task_id

def main():

  # 已經完成 fan-in 的 Evidence
  current = [
    {
      "task_id": "task_1",
      "answer": "Old TSMC answer",
    },
    {
      "task_id": "task_2",
      "answer": "MediaTek answer",
    }
  ]

  # Answer Regenerator 修正 task_1
  updates = [
    {
      "task_id": "task_1",
      "answer": "Corrected TSMC answer",
    },
  ]

  result = merge_evidence_by_task_id(
    current,
    updates,
  )

  print("\nEvidence after update:")

  for item in result:
    print(item)

  # Evidence 總數不能增加
  assert len(result) == 2

  evidence_map = {
    item["task_id"]: item
    for item in result
  }

  # task_1 必須更新
  assert (
    evidence_map["task_1"]["answer"]
    == "Corrected TSMC answer"
  )

  # task_2 必須保持原狀
  assert(
    evidence_map["task_2"]["answer"]
    == "MediaTek answer"
  )

  # 測試新增 task_3
  result = merge_evidence_by_task_id(
    result,
    [
      {
        "task_id": "task_3",
        "answer": "New evidence",
      }
    ]
  )

  assert len(result) == 3

  # 再次更新 task_1，數量仍然不能增加
  result = merge_evidence_by_task_id(
    result,
    [
      {
        "task_id": "task_1",
        "answer": "Final TSMC answer",
      }
    ]
  )

  assert len(result) == 3

  assert (
    next(
      item
      for item in result
      if item["task_id"] == "task_1"
    )["answer"]
    == "Final TSMC answer"
  )

  print(
    "\n[PASS] Evidence reducer tests passed."
  )

  from langgraph.graph import StateGraph, START, END
  from agent_state import FinancialResearchState

  def update_evidence(state):
    return {
      "evidence": [
        {
          "task_id": "task_1",
          "answer": "Corrected answer",
        }
      ]
    }

  graph = StateGraph(FinancialResearchState)
  graph.add_node(
    "update_evidence",
    update_evidence,

  )

  graph.add_edge(
    START,
    "update_evidence",
  )

  graph.add_edge(
    "update_evidence",
    END,
  )

  app = graph.compile()

  result = app.invoke({
    "evidence": [
      {
        "task_id": "task_1",
        "answer": "Old answer",
      },
      {
        "task_id": "task_2",
        "answer": "Unchanged answer"
      }
    ]
  })

  assert len(result['evidence']) == 2
  assert next(
    item for item in result["evidence"]
    if item["task_id"] == "task_1"
  )["answer"] == "Corrected answer"

  print(
    "[PASS] LangGraph Evidence update passed."
  )

if __name__ == "__main__":
  main()