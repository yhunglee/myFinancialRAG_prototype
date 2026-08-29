from __future__ import annotations

from agent_node import intent_router

def main():

  def gen_state(question, current_task_num):
    return  {
      "question": question,
      "intent": "",
      "companies": [],
      "periods": [],
      "router_confidence": 0.0,
      "research_plan": [],
      "current_task": current_task_num,
      "evidence": [],
      "sufficient": False,
      "missing_information": [],
      "final_answer": "",
    }

  state0 = gen_state('台積電 2025 年第四季營收是多少？', 0)
  state1 = gen_state('比較台積電和聯發科 2025 年第四季營收', 1)
  state2 = gen_state('研究 AI 晶片產業上下游公司', 2)
  state3 = gen_state('什麼是毛利率？', 3)

  states = [state0, state1, state2, state3]
  for state in states:
    result = intent_router(state)
    print(f'question: {state["question"]}')
    print("Router result:")
    print(result)
    print('=======')


if __name__ == '__main__':
  main()