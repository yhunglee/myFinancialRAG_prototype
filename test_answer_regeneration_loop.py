from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_node import (
  evidence_checker,
  answer_regenerator,
)

from agent_graph import (
  route_after_evidence_check,
)

def build_test_graph():

  graph = StateGraph(
    FinancialResearchState
  )

  graph.add_node(
    "evidence_checker",
    evidence_checker,
  )

  graph.add_node(
    "answer_regenerator",
    answer_regenerator,
  )

  graph.add_edge(
    START,
    "evidence_checker"
  )

  graph.add_conditional_edges(
    "evidence_checker",
    route_after_evidence_check,
    {
      "proceed": END,
      "retrieve_again": END,
      "regenerate_answer": "answer_regenerator",
      "stop": END,
    }
  )

  graph.add_edge(
    "answer_regenerator",
    "evidence_checker",
  )

  return graph.compile()


def main():
  graph = build_test_graph()

  initial_state: FinancialResearchState = {
    "question": "Retrieve the revenue for TSMC in Q4 2025.",
    "intent": "single_company",
    "companies": ["TSMC"],
    "periods": ["2025Q4"],
    "router_confidence": 1.0,
    "research_plan": [
      {
        "task_id": "task_1",
        "company": "TSMC",
        "period": "2025Q4",
        "topic": "營收",
        "query":
          "Retrieve the revenue for TSMC in Q4 2025.",
      }
    ],
    "current_task": 1,

    # ------------------------------
    # 故意製造錯誤答案
    #
    # source:
    #   NT$ 1,046.09 billion
    #
    # wrong answer:
    #   1,046.09 億
    #
    # 1 billion = 10 億
    # 所以這個答案單位轉換錯誤
    # ------------------------------
    "evidence": [
      {
        "task_id": "task_1",

        "company": "TSMC",

        "period": "2025Q4",

        "query": "Retrieve the revenue for TSMC in Q4 2025.",

        "answer": "TSMC Q4 2025 revenue was 1,046.09 億元。",

        "retrieved_contexts": [
          """
          TSMC Fourth Quarter 2025 Results

          (In NT$ billions)

          Net Revenue | 1,046.09
          """
        ],

        "metadata": [
          {
            "ticker": "2330",
            "market": "TW",
            "year": 2025,
            "quarter": "Q4",
            "chunk_index": 0,
          }
        ],

        "sources": [
          {
            "ticker": "2330",
            "market": "TW",
            "year": 2025,
            "quarter": "Q4",
            "chunk_index": 0,
          }
        ],
      }
    ],

    "sufficient": False,
    "missing_information": [],
    "weak_evidence": [],
    "unsupported_answer": [],
    "failure_type": "none",
    "next_action": "proceed",
    "regeneration_count": 0,
    "final_answer": ""
  }

  result = graph.invoke(
    initial_state
  )

  print("\n========================")
  print("Final Graph Reshult")
  print("==========================")

  print(
    "next_action: ",
    result["next_action"]
  )

  print(
    "regeneration_count: ",
    result["regeneration_count"]
  )

  print(
    "failure_type:",
    result["failure_type"]
  )

  print(
    "sufficient:",
    result["sufficient"],
  )

  print("\nFinal Evidence Answer:")

  for item in result["evidence"]:
    print(item["answer"])


if __name__ == '__main__':
  main()