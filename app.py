from __future__ import annotations

"""
Chainlit 的主要檔案
"""
import chainlit as cl

import asyncio

from agent_graph import build_agent_graph

agent_graph = build_agent_graph()

def create_initial_state(question: str) -> dict:
  return {
    "question": question,

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

  await cl.Message(
    content=(
      "Financial Agentic RAG 已啟動。\n\n"
      "你可以詢問財報資訊，例如:\n"
      "`比較台積電和聯發科 2025 年第四季營收`"
    )
  ).send()

@cl.on_message
async def main(message: cl.Message):

  # 一開始就建立並發送思考過程
  think_step = cl.Step(name="AI 財報推理過程", type="run")
  await think_step.send()

  msg = None
  async for event in rag_service.rag_chat_stream(message.content):
    if event.type == "reasoning":
        await think_step.stream_token(event.content)

    if event.type == "answer":
      if not msg:
        msg = cl.Message(content="", parent_id=think_step.id)
        await msg.send()
      
      await msg.stream_token(event.content)

  await think_step.update()

  if msg:
    await msg.update()


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")