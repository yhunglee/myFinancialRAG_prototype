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

@cl.on_message
async def main(message: cl.Message):
  # Your custom logic goes here...

  msg = cl.Message(content="")
  think_step = None
  is_thinking = False

  # 建立緩衝區來處理被切斷的標籤字串
  buffer = ''
  for token in rag_service.rag_chat_stream(message.content):
    buffer += token

    # 判斷是否進入思考區塊
    if "<think>" in buffer and not is_thinking:
      is_thinking = True
      buffer = buffer.replace("<think>", "") # 清除標籤
      think_step = cl.Step(name="AI財報推理過程", type="run")
      await think_step.send()

    # 判斷是否結束思考區塊
    if "</think>" in buffer and is_thinking:
      is_thinking = False

      # 處理掉 </think> 之前的剩餘思考文字
      thought_text = buffer.split("</think>")[0]
      if thought_text:
        await think_step.stream_token(thought_text)
      await think_step.update()

      # 將 buffer 剩下的解答部分保留
      buffer = buffer.split("</think>")[1]
      await msg.send() # 開始發送主訊息

    # 根據狀態決定 Token 要流向哪裡
    if is_thinking and think_step:
      # 清空緩衝區並輸出到思考面板
      await think_step.stream_token(buffer)
      buffer = ''
    elif not is_thinking and not buffer.isspace():
      # 清空緩衝區並輸出到主畫面
      await msg.stream_token(buffer)
      buffer = ''

  # 迴圈結束後，更新最終 UI 狀態
  if think_step:
    await think_step.update()

  await msg.update()

    

  # ======
  # response = rag_service.rag_chat(message.content)

  # # Send a response back to the user
  # await cl.Message(
  #     content=f"{response}",
  # ).send()

@cl.on_chat_end
def on_chat_end():
  rag_service.clean_history()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")