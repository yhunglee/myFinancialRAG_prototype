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

  # 一開始就建立並發送主對話框
  msg = cl.Message(content="")
  await msg.send()

  think_step = None
  is_thinking = False

  # 建立緩衝區來處理被切斷的標籤字串
  buffer = ''

  # 用來比對是否正在形成標籤的字首
  target_tags = ["<think>", "</think>"]

  async for token in rag_service.rag_chat_stream(message.content):
    buffer += token

    # 1. 判斷是否完整捕捉到標籤開始
    if "<think>" in buffer:
      is_thinking = True
      buffer = buffer.replace("<think>", "") # 清除標籤
      if not think_step:
        # 1-1. 利用 parent_id=msg.id 將思考面板強制掛載到主對話框內部
        think_step = cl.Step(name="AI 財報推理過程", type="run", parent_id=msg.id)
        await think_step.send()

    # 2. 判斷是否完整捕捉到標籤結束
    if "</think>" in buffer:
      is_thinking = False
      parts = buffer.split("</think>")
      thought_text = parts[0]

      # 把結束標籤前的文字送進思考面板
      if thought_text and think_step:
        await think_step.stream_token(thought_text)
      if think_step:
        await think_step.update()

      # 剩下的文字保留給主畫面
      buffer = parts[1] if len(parts) > 1 else ""

    # 3. 核心邊界處理: 檢查緩衝區結尾是否為標籤的[部分片段]
    is_forming_tag = False
    for tag in target_tags:
      # 檢查從長度 1 到 tag 全長， buffer 的結尾是否吻合
      for i in range(1, len(tag)):
        if buffer.endswith(tag[:i]):
          is_forming_tag = True
          break
      if is_forming_tag:
        break

    # 4. 如果沒有正在形成標籤，才安全地將緩衝區內容輸出
    if not is_forming_tag:
      if is_thinking and think_step:
        await think_step.stream_token(buffer)
        buffer = ""
      elif not is_thinking:
        # 3. 直接輸出至主對話框，再也不用擔心順序錯亂
        if buffer:
          await msg.stream_token(buffer)
          buffer = ""

  # 5. 確保迴圈結束後，剩餘的字元也有輸出
  if buffer:
    if is_thinking and think_step:
      await think_step.stream_token(buffer)
    elif not is_thinking:
      await msg.stream_token(buffer)

  # 迴圈結束後，更新最終 UI 狀態
  if think_step:
    await think_step.update()
  await msg.update()


@cl.on_chat_end
def on_chat_end():
  rag_service.clean_history()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")