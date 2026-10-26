
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_node import (
  has_financial_numeric_answer,
  task_evidence_checker,
)


def test_missing_numeric_answer_is_rejected():

  state = {
    "task": {
      "task_id": "task_1",
      "company": "台積電",
      "period": "2025Q4",
      "topic": "營收",
      "query": "Retrieve TSMC revenue in 2025Q4",
    },
    "task_evidence": {
      "answer": "財報未揭露",
      "retrieved_contexts": [
          (
              "(In NT$ billions) | 4Q25\n"
              "Net Revenue | 1,046.09"
          )
      ],
    },
    "retrieval_count": 0,
  }

  result = task_evidence_checker(state)

  assert result["sufficient"] is False
  assert result["failure_type"] == "weak_evidence"
  assert result["next_action"] == "retrieve_again"


def test_numeric_detection():

  assert has_financial_numeric_answer(
    "台積電 2025Q4 營收為 1,046.09 billion"
  ) is True

  assert has_financial_numeric_answer(
    "台積電 2025Q4 營收財報未揭露"
  ) is False

  assert has_financial_numeric_answer(
    "聯發科營收為 150,188 百萬元"
  ) is True

  assert has_financial_numeric_answer(
    "2025 年第四季財報未揭露"
  ) is False
