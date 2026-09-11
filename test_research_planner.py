from __future__ import annotations

from agent_node import research_planner

def main():
  state = {
    "question": "比較台積電和聯發科 2025 年第四季營收",
    "intent": "peer_comparison",
    "companies": ["台積電", "聯發科"],
    "periods": ["2025Q4"],
    "router_confidence": 1.0,
    "research_plan": [],
    "current_task": 0,
    "evidence": [],
    "sufficient": False,
    "missing_information": [],
    "final_answer": "", 
  }

  result = research_planner(state)
  print(result)

if __name__ == '__main__':
  main()