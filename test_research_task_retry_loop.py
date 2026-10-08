from __future__ import annotations

import asyncio
from unittest.mock import patch


async def main():

  """
  這個 testcase 驗證：

  attempt 1
      ↓
  insufficient
      ↓
  prepare_task_retry
      ↓
  attempt 2
      ↓
  sufficient
      ↓
  finalize_task_evidence
      ↓
  END

  預期：
  retrieval_count == 1
  evidence count == 1

  代表第一次失敗的 evidence
  不會被累積到最終 evidence。
  """


  async def fake_retrieve_task(state):
    """
    模擬 retrieval。

    第一次 retrieval_count = 0：
      回傳一份「失敗版本」 evidence。

    第二次 retrieval_count = 1：
      回傳一份「成功版本」 evidence。
    """

    retrieval_count = state.get(
      "retrieval_count",
      0,
    )

    task = state["task"]

    print(
      "[TEST] retrieve_task:",
      f"retrieval_count={retrieval_count}"
    )

    if retrieval_count == 0:

      # 第一次故意產生不足的 evidence
      return {
        "task_evidence": {
          "task_id": task["task_id"],
          "company": task["company"],
          "period": task["period"],
          "topic": task["topic"],
          "query": task["query"],

          "answer": (
            "Insufficient evidence."
          ),

          "retrieved_contexts": [],

          "metadata": [],

          "sources": [],
        }
      }

    # 第二次模擬成功
    return {
      "task_evidence": {
        "task_id": task["task_id"],
        "company": task["company"],
        "period": task["period"],
        "topic": task["topic"],
        "query": task["query"],

        "answer": (
          "TSMC Fourth Quarter 2025 "
          "Net Revenue was "
          "NT$1,046.09 billion."
        ),

        "retrieved_contexts": [
          (
            "Fourth Quarter 2025 "
            "Net Revenue: "
            "NT$1,046.09 billion."
          )
        ],

        "metadata": [
          {
            "ticker": "2330",
            "year": 2025,
            "quarter": "Q4",
          }
        ],

        "sources": [
          {
            "ticker": "2330",
            "year": 2025,
            "quarter": "Q4",
          }
        ],
      }
    }


  def fake_task_evidence_checker(state):
    """
    第一次故意判定 insufficient。
    第二次判定 sufficient。

    使用 retrieval_count 來判斷目前是第幾次。
    """

    retrieval_count = state.get(
      "retrieval_count",
      0,
    )

    print(
      "[TEST] task_evidence_checker:",
      f"retrieval_count={retrieval_count}"
    )

    if retrieval_count == 0:

      print(
        "[TEST] Force insufficient "
        "on first attempt."
      )

      return {
        "sufficient": False,
        "failure_type": "missing_evidence",
        "next_action": "retrieve_again",
      }

    print(
      "[TEST] Force sufficient "
      "on second attempt."
    )

    return {
      "sufficient": True,
      "failure_type": "none",
      "next_action": "proceed",
    }


  """
  注意：
  patch 必須在 import / build graph 前生效。
  """

  with patch(
    "research_task_graph.retrieve_task",
    new=fake_retrieve_task,
  ), patch(
    "research_task_graph.task_evidence_checker",
    new=fake_task_evidence_checker,
  ):

    from research_task_graph import (
      build_research_task_graph,
    )

    graph = build_research_task_graph()

    initial_state = {
      "task": {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "topic": "營收",
        "query": "台積電 2025年第四季營收",
      },

      "task_evidence": None,

      "sufficient": False,

      "failure_type": "none",

      "next_action": "proceed",

      "retrieval_count": 0,

      "evidence": [],
    }

    result = await graph.ainvoke(
      initial_state
    )


  print(
    "\n=========================="
  )

  print(
    "Retry Loop Test Result"
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

  print(
    "final task answer:",
    result["task_evidence"]["answer"]
  )


  # ==========================
  # Assertions
  # ==========================

  assert result["sufficient"] is True

  """
  第一次失敗後只 retry 一次，
  所以 retrieval_count 必須是 1。
  """
  assert result["retrieval_count"] == 1

  """
  即使 retrieval 執行兩次，
  最終 evidence 只能有一筆。

  第一次失敗 Evidence
  不可以進入 fan-in collection。
  """
  assert len(result["evidence"]) == 1

  assert (
    result["evidence"][0]["answer"]
    ==
    result["task_evidence"]["answer"]
  )

  assert (
    "1,046.09"
    in result["evidence"][0]["answer"]
  )

  print(
    "\n[PASS] Task-level retry loop "
    "executed successfully."
  )


if __name__ == "__main__":
  asyncio.run(main())