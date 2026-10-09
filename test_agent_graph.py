from __future__ import annotations

from agent_graph import build_agent_graph
import asyncio


async def main():
  graph = build_agent_graph()

  initial_state = {
    "question": "比較台積電和聯發科 2025 年第四季營收",
    "standalone_question": "",
    "chat_history": [],

    "intent": "",
    "companies": [],
    "periods": [],
    "router_confidence": 0.0,

    "research_plan": [],
    "current_task": 0,

    "previous_validated_evidence": [],
    "reuse_evidence": False,

    "evidence": [],

    "sufficient": False,
    "missing_information": [],
    "weak_evidence": [],
    "unsupported_answer": [],

    "failure_type": "none",
    "next_action": "proceed",

    "regeneration_count": 0,
    "retrieval_count": 0,

    "calculation_required": False,
    "calculation_results": [],
    
    "final_answer": "",
  }

  result = await graph.ainvoke(initial_state)

  print("Graph Result:")
  print(result)

if __name__ == "__main__":
  asyncio.run(main())