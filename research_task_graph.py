from __future__ import annotations

from langgraph.graph import (
  StateGraph,
  START,
  END,
)

from agent_state import ResearchTaskState
from agent_node import retrieve_task

def bulid_search_task_again():
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

  graph.add_edge(
    START,
    "retrieve_task",
  )

  graph.add_edge(
    "retrieve_task",
    END,
  )

  return graph.compile()

