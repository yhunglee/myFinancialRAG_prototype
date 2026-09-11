from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from agent_state import FinancialResearchState
from agent_node import (
  evidence_checker,
  retrieve_again,
)

from agent_graph import (
  route_after_evidence_check,
)

def build_test_graph():

  graph = StateGraph(
    FinancialResearchState,
  )

  graph.add_node(
    "evidence_checker",
    evidence_checker
  )

  graph.add_node(
    "retrieve_again",
    retrieve_again,
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
      "retrieve_again": "retrieve_again",
      "regenerate_answer": END,
      "stop": END,
    }
  )

  graph.add_edge(
    "retrieve_again",
    "evidence_checker"
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
    # 故意製造[沒有 Evidence] 的狀態
    #
    # Evidence_checker() 看到空 evidence 
    # 應該會:
    # failure_type = missing_evidence
    # next_action = retrieve_again
    # 
    # 然後 Graph 執行 retrieve_again()
    # ------------------------------
    "evidence": [
    ],

    "sufficient": False,
    "missing_information": [],
    "weak_evidence": [],
    "unsupported_answer": [],
    "failure_type": "none",
    "next_action": "proceed",
    "regeneration_count": 0,
    "retrieval_count": 0,
    "final_answer": ""
  }

  result = graph.invoke(initial_state)

  assert result["retrieval_count"] == 1, (
    "retrieve_again node was not executed exactly once."
  )

  assert len(result["evidence"]) > 0, (
    "retrieve_again did not produce any evidence."
  )

  print("\n[PASS] retrieve_again loop executed successfully.")

  print("\n========================")
  print("Final Graph Result")
  print("==========================")
  
  print(
    "next_action: ",
    result["next_action"]
  )

  print(
    "retrieval_count: ",
    result["retrieval_count"]
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