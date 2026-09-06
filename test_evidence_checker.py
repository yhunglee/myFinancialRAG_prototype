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


def test_answer_not_supported():

  state = {
    "question": "Retrieve the revenue for 台積電 in 2025Q4.",

    "research_plan": [
      {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "topic": "營收",
        "query": "Retrieve the revenue for 台積電 in 2025Q4.",
      }
    ],

    "evidence": [
      {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "query": "Retrieve the revenue for 台積電 in 2025Q4.",

        # 故意寫錯單位
        "answer": "台積電 2025 年第四季營收為新台幣 1,046.09 億元。",

        # context 明確寫 billion
        "retrieved_contexts": [
            """
            (In NT$ billions)
            Net Revenue | 4Q25
            1,046.09
            """
        ],

        "metadata": [
          {
            "ticker": "2330",
            "market": "TW",
            "year": 2025,
            "quarter": "Q4",
            "chunk_index": 25,
          }
        ],

        "sources": [
          {
            "ticker": "2330",
            "market": "TW",
            "year": 2025,
            "quarter": "Q4",
            "chunk_index": 25,
          }
        ],
      }
    ],
  }

  result = evidence_checker(state)

  print("Answer consistency test:")
  print(result)

if __name__ == "__main__":
  # main()

  test_answer_not_supported()
