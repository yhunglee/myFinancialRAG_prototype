from __future__ import annotations

from agent_graph import (
  route_after_global_evidence_check,
)

def main():

  # Case A: 第一次需要修正
  state = {
    "sufficient": False,
    "next_action": "regenerate_answer",
    "regeneration_count": 0,
  }

  assert (
    route_after_global_evidence_check(state)
    == "regenerate_answer"
  )

  print(
    "[PASS] First regeneration allowed."
  )

  # case B: 第二次仍允許修正
  state["regeneration_count"] = 1

  assert (
    route_after_global_evidence_check(state)
    == "regenerate_answer"
  )

  print("[PASS] Second regeneration allowed")

  # Case C: 達到上限
  state['regeneration_count'] = 2

  assert (
    route_after_global_evidence_check(state)
    == "stop"
  )

  # Case D: 修正成功
  state["sufficient"] = True
  state["next_action"] = "proceed"

  assert (
    route_after_global_evidence_check(state)
    == "proceed"
  )

  # Case E: 全域重新檢索仍然停用
  state["sufficient"] = False
  state["next_action"] = "retrieve_again"

  assert (
    route_after_global_evidence_check(state)
    == "stop"
  )

  print("[PASS] Global retrieval remains disabled")

if __name__ == '__main__':
  main()