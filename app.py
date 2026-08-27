from __future__ import annotations

"""
Chainlit 的主要檔案
"""
import chainlit as cl

from myrag_module import FinancialRAGService


rag_service = FinancialRAGService(
  db_path='./chroma_db',
  collection_name="financial_reports"
)

@cl.on_chat_start
def on_chat_start():
  print("A new chat session has started!")
  rag_service.clean_history()

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
  


@cl.on_chat_end
def on_chat_end():
  rag_service.clean_history()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")