from __future__ import annotations

from agent_node import evidence_checker

def main():
  state = {
    "question": "比較台積電和聯發科 2025 年第四季營收",
    "research_plan": [
      {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "topic": "營收",
        "query": "台積電 2025年第四季營收",
      },
      {
        "task_id": "task_2",
        "company": "聯發科",
        "period": "2025Q4",
        "topic": "營收",
        "query": "聯發科 2025年第四季營收",
      },
    ],
    "evidence": [
      {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "query": "台積電 2025年第四季營收",
        "answer": "台積電 2025 年第四季營收為NT$1,046.09 billion",
        "retrieved_contexts": [
            "TSMC fourth quarter revenue was NT$1,046.09 billion"
        ],
        "metadata": [],
      },
      {
        "task_id": "task_2",
        "company": "聯發科",
        "period": "2025Q4",
        "query": "聯發科 2025年第四季營收",
        "answer": "聯發科 2025 年第四季營收為NT$150,188 百萬元",
        "retrieved_contexts": [
            "聯發科2025年第四季營收是 NT$150,188 百萬元"
        ],
        "metadata": [],
      }
    ],
  }

  result = evidence_checker(state)

  print("Evidence Check Result:")
  print(result)

if __name__ == "__main__":
  main()
