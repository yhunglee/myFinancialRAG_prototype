from __future__ import annotations

from langgraph.types import Send

from agent_state import FinancialResearchState
from research_task_graph import build_research_task_graph

def create_task_state(task: dict) -> dict:
  """
  為每個 ResearchTask 建立獨立的初始 State
  """

  return {
    "task": task,
    "task_evidence": None,
    "sufficient": False,
    "failure_type": "none",
    "next_action": "proceed",
    "retrieval_count": 0,
    "evidence": [],
  }

def dispatch_research_tasks(
  state: FinancialResearchState,
):
  """
  動態 Fan-out。

  Evidence Reuse 命中時:
    直接進 Calculator

  沒有命中時:
    每個 ResearchTask 各自透過 Send 執行。
  """

  if state.get("reuse_evidence", False):
    return "calculator"

  tasks = state.get("research_plan", [])

  if not tasks:
    return "failure_report_writer"

  return [
    Send(
      "research_task_worker",
      create_task_state(task),
    )
    for task in tasks
  ]

async def research_task_worker(state: dict) -> dict:
  """
  執行一個完整的 ResearchTask Subgraph

  每個 Send 都會啟動獨立的執行，
  因此具有自己的 retry counter。
  """

  task = state["task"]

  subgraph = build_research_task_graph()

  result = await subgraph.ainvoke(state)

  sufficient = result["sufficient"]

  task_result = {
    "task_id": task["task_id"],
    "company": task.get("company", ""),
    "period": task.get("period", ""),
    "topic": task.get("topic", ""),
    "sufficient": sufficient,
    "retrieval_count": result["retrieval_count"],
    "failure_type": result["failure_type"],
  }

  print(
    "[Research Task Completed]",
    task_result,
  )

  # 只有驗證成功的證據，可以進入 parent state
  validated_evidence = (
    result.get("evidence", [])
    if sufficient
    else []
  )

  return {
    "evidence": validated_evidence,
    "task_results": [task_result],
  }


def fan_in_gate(
  state: FinancialResearchState,
) -> dict:
  """
  等全部 ResearchTask 完成後，
  檢查是否每個必要任務都成功。
  """

  plan = state.get("research_plan", [])
  results = state.get("task_results", [])
  evidence = state.get("evidence", [])

  expected_ids = {
    task["task_id"]
    for task in plan
  }

  result_ids = [
    item["task_id"]
    for item in results
  ]

  valid_ids = {
    item["task_id"]
    for item in results
    if item["sufficient"]
  }

  evidence_ids = [
    item["task_id"]
    for item in evidence
  ]

  """
  研究計畫非空
  結果數量正確
  結果 ID 正確
  Evidence 數量正確
  Evidence ID 正確
  全部任務驗證成功
  """
  all_successful = (
    bool(expected_ids)
    and len(result_ids) == len(expected_ids)
    and set(result_ids) == expected_ids
    and len(evidence_ids) == len(expected_ids)
    and set(evidence_ids) == expected_ids
    and valid_ids == expected_ids
  )

  if all_successful:
    print(
      "[Fan-in] All research tasks succeeded."
    )

    return {
      "sufficient": True,
      "failure_type": "none",
    }

  failed= [
    item for item in results
    if not item["sufficient"]
  ]

  print(
    "[Fan-in] Incomplete research tasks:",
    failed,
  )

  """
  Notice:
  這裡的 missing_evidence 是整份研究未完成的概括狀態，個別 task
  仍可能是 weak_evidence
  """
  return {
    "sufficient": False,
    "failure_type": "missing_evidence",
  }

def route_after_fan_in(
  state: FinancialResearchState,
) -> str:

  if state["sufficient"]:
    return "proceed"

  return "stop"