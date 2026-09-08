from __future__ import annotations

"""
Chainlit 的主要檔案
"""
import chainlit as cl

import asyncio

from agent_graph import build_agent_graph

agent_graph = build_agent_graph()

def create_initial_state(
    question: str,
    chat_history: list[dict],
  ) -> dict:
  return {
    "question": question,

    "standalone_question": "",
    
    "chat_history": chat_history,

    # intent_router
    "intent": "",
    "companies": [],
    "periods": [],
    "router_confidence": 0.0,

    # research_planner
    "research_plan": [],

    # rag_executor
    "current_task": 0,
    "evidence": [],

    # evidence_checker
    "sufficient": False,
    "missing_information": [],
    "weak_evidence": [],
    "unsupported_answer": [],
    "failure_type": "none",
    "next_action": "proceed",

    # retry loop
    "regeneration_count": 0,
    "retrieval_count": 0,

    # report_writer
    "final_answer": "",
  }

@cl.on_chat_start
async def on_chat_start():

  cl.user_session.set(
     "chat_history",
     [],
  )

  await cl.Message(
    content=(
      "Financial Agentic RAG 已啟動。\n\n"
      "你可以詢問財報資訊，例如:\n"
      "`比較台積電和聯發科 2025 年第四季營收`"
    )
  ).send()

@cl.on_message
async def main(message: cl.Message):

  initial_state = create_initial_state(
     message.content,
  )

  """
  Notice: 因為 LangGraph invoke() 是同步工作，
  所以放到 thread 執行，避免 blocking Chainlit event loop
  """
  result = await asyncio.to_thread(
     agent_graph.invoke,
     initial_state,
  )

  final_answer = result.get(
     "final_answer",
     "",
  )

  await cl.Message(
     final_answer
  ).send()


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")