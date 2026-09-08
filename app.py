from __future__ import annotations

"""
Chainlit 的主要檔案
"""
import chainlit as cl

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


STEP_NAMES = {
  "contextualize_question": "Resolving Conversation Context",
  "intent_router": "Understanding Question",
  "research_planner": "Planning Research",
  "rag_executor": "Retrieving Evidence",
  "evidence_checker": "Checking Evidence",
  "retrieve_again": "Retrieving Additional Evidence",
  "answer_regenerator": "Regenerating Answer",
  "report_writer": "Writing Report",
  "failure_report_writer": "Writing Failure Report"   
}

def format_step_output(
  node_name: str,
  update: dict,
) -> str:
  """
  將 LangGraph raw state 轉換成一般使用者看的內容
  """

  if node_name == "contextualize_question":
    return (
      f"Standalone question:\n"
      f"{update.get('standalone_question', '')}"
    )

  if node_name == "intent_router":
    return (
      f"Intent: {update.get('intent')}\n\n"
      f"Companies: {update.get('companies')}\n\n"
      f"Periods: {update.get('periods')}\n\n"
      f"Confidence: {update.get('router_confidence')}"
    )

  if node_name == "research_planner":
    research_plan = update.get(
      "research_plan",
      []
    )

    if not research_plan:
      return "No research tasks generated."

    lines = []

    for task in research_plan:
      lines.append(
        "\n".join([
          f"Task: {task.get('task_id')}",
          f"Company: {task.get('company')}",
          f"Period: {task.get('period')}",
          f"Topic: {task.get('topic')}",
          f"Query: {task.get('query')}"
        ])
      )

    return "\n\n".join(lines)

  if node_name in {
    "rag_executor",
    "retrieve_again",
  }:
    evidence = update.get(
      "evidence",
      [],
    )

    lines = [
      f"Evidence items: {len(evidence)}"
    ]

    if node_name == "retrieve_again":
      lines.append(
        f"Retrieve count: "
        f"{update.get('retrieval_count', 0)}"
      )

    for item in evidence:
      sources = item.get(
        "sources",
        [],
      )

      lines.append(
        "\n".join([
          "",
          f"Task: {item.get('task_id')}",
          f"Company: {item.get('company')}",
          f"Period: {item.get('period')}",
          f"Query: {item.get('query')}",
          f"Retrieved sources: {len(sources)}",
          f"Answer: {item.get('answer')}",
        ])
      )

    return "\n".join(lines)

  if node_name == "evidence_checker":
    return (
      f"Sufficient: {update.get('sufficient')}\n\n"
      f"Failure type: {update.get('failure_type')}\n\n"
      f"Next action: {update.get('next_action')}\n\n"
      f"Missing information: "
      f"{update.get('missing_information', [])}\n\n"
      f"Weak evidence: "
      f"{update.get('weak_evidence', [])}\n\n"
      f"Unsupported answer: "
      f"{update.get('unsupported_answer', [])}"
    )

  if node_name == "answer_regenerator":
    evidence = update.get(
      "evidence",
      []
    )

    return (
      f"Regeneration count: "
      f"{update.get('regeneration_count', 0)}\n\n"
      f"Updated evidence items: {len(evidence)}"
    )

  if node_name in {
    "report_writer",
    "failure_report_writer",
  }:
    return update.get(
      "final_answer",
      ""
    )

  return str(update)

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

  chat_history = cl.user_session.get(
    "chat_history"
  ) or []

  initial_state = create_initial_state(
    question=message.content,
    chat_history=chat_history,
  )

  final_state = dict(initial_state)

  parent_step = cl.Step(
    name="Agent Research Process",
    type="tool",
    default_open=False,
  )

  parent_step.output = "Running..."
  await parent_step.send()

  """
  紀錄目前正在執行中的 Chainlit Step。

  key:
    LangGraph event 的 run_id

  value:
    對應的 Chainlit Step
  """
  active_steps = {}

  async for event in agent_graph.astream_events(
    initial_state,
    version="v2",
    include_names=list(STEP_NAMES.keys())
  ):

    event_type = event["event"]
    node_name = event["name"]
    run_id = event["run_id"]

    """
    Node 開始執行
    """
    if event_type == 'on_chain_start':

      step_name = STEP_NAMES.get(
        node_name,
        node_name,
      )

      step = cl.Step(
        name=step_name,
        type="tool",
        parent_id=parent_step.id,
        default_open=False,
      )

      step.input = node_name
      step.output = "Running..."

      await step.send()

      active_steps[run_id] = step
      continue

    """
    Node 執行完成
    """
    if event_type == 'on_chain_end':

      update = event.get(
        "data",
        {}
      ).get(
        "output",
        {}
      )

      if not isinstance(update, dict):
        continue

      final_state.update(update)

      step = active_steps.pop(
        run_id,
        None,
      )

      if step is None:
        continue

      step.output = format_step_output(
        node_name=node_name,
        update=update,
      )

      await step.update()

  parent_step.output = "Research process completed."
  await parent_step.update()

  final_answer = final_state["final_answer"]

  await cl.Message(
    content=final_answer,
  ).send()

  chat_history.append({
    "role": "user",
    "content": message.content,
  })

  chat_history.append({
    "role": "assistant",
    "content": final_answer,
  })

  cl.user_session.set(
    "chat_history",
    chat_history
  )

  if final_state.get("sufficient") is True:
    cl.user_session.set(
      "previous_validated_evidence",
      final_state.get("evidence", [])
    )


@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")