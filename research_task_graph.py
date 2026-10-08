from __future__ import annotations

from langgraph.graph import (
  StateGraph,
  START,
  END,
)

from agent_state import ResearchTaskState
from agent_node import (
  retrieve_task,
  task_evidence_checker,
  prepare_task_retry,
  finalize_task_evidence,
)

MAX_TASK_RETRIEVAL_ATTEMPTS = 2

def build_research_task_graph():
  """
  建立單一 ResearchTask 使用的 Subgraph。

  START
   ↓
  retrieve_task
    ↓
    END

  TODO: 
  task_evidence_checker
  retrieve_again
  conditional_retry
  
  """

  graph = StateGraph(
    ResearchTaskState
  )

  graph.add_node(
    "retrieve_task",
    retrieve_task,
  )

  

  graph.add_node(
    "task_evidence_checker",
    task_evidence_checker,
  )

  graph.add_node(
    "prepare_task_retry",
    prepare_task_retry,
  )

  graph.add_node(
    "finalize_task_evidence",
    finalize_task_evidence,
  )

  graph.add_edge(
    START,
    "retrieve_task",
  )

  graph.add_edge(
    "retrieve_task",
    "task_evidence_checker",
  )

  graph.add_conditional_edges(
    "task_evidence_checker",
    route_after_task_evidence_check,
    {
      "proceed": "finalize_task_evidence",
      "retrieve_again": "prepare_task_retry",
      "stop": END
    }
  )

  graph.add_edge(
    "prepare_task_retry",
    "retrieve_task",
  )

  graph.add_edge(
    "finalize_task_evidence",
    END,
  )

  return graph.compile()


def route_after_task_evidence_check(
  state: ResearchTaskState,
) -> str:

  next_action = state['next_action']
  if next_action == 'proceed':
    return "proceed"

  if next_action == 'retrieve_again':
    retrieval_count = state.get(
      "retrieval_count",
      0,
    )

    if retrieval_count >= MAX_TASK_RETRIEVAL_ATTEMPTS:
      return "stop"

    return "retrieve_again"

  return "stop"