from __future__ import annotations

from agent_graph import build_agent_graph

def main():
  graph = build_agent_graph()

  initial_state = {
    "question": "比較台積電和聯發科 2025 年第四季營收",
    "intent": "",
    "companies": [],
    "periods": [],
    "router_confidence": 0.0,
    "research_plan": [],
    "current_task": 0,
    "evidence": [],
    "sufficient": False,
    "missing_information": [],
    "final_answer": "",
  }

  result = graph.invoke(initial_state)

  print("Graph Result:")
  print(result)

if __name__ == "__main__":
    main()