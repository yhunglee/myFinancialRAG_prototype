from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

from agent_state import FinancialResearchState
from myrag_module import FinancialRAGService
import re

FINANCIAL_NUMBER_PATTERN = r"\d+(?:,\d{3})*(?:\.\d+)?"

client = OpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio"
)

MODEL_NAME = 'local_model'

rag_service = FinancialRAGService(
  db_path="./chroma_db",
  collection_name="financial_reports",
)

class RouterResult(BaseModel):
  """
  自然語言問題, 轉成程式可以理解的決策
  """
  intent: Literal[
    "single_company",
    "peer_comparison",
    "industry_research",
    "general_financial_question"
  ]

  companies: list[str]
  periods: list[str]
  confidence: float

def contextualize_question(
  state: FinancialResearchState,
) -> dict:
  question = state["question"]
  chat_history = state.get(
    "chat_history",
    []
  )

  """
  第一輪沒有對話歷史，
  不需要浪費一次 LLM
  """
  if not chat_history:
    return {
      "standalone_question": question
    }

  system_prompt = """
  You are a conversation contextualization component for 
  a financial research assistant.

  Your task is to rewrite the user's current question
  into a standalone question that can be understood without
  seeing the previous conversation.

  Rules:

  1. Use the conversation history only to resolve context.
  2. Preserve the user's current intent.
  3. Resolve references such as:
     - 他們
     - 那家公司
     - 上一季
     - 前面提到的公司
     - the two companies
     - that company
     - previous quarter
  4. Do not answer the question.
  5. Do not add financial facts that are not present 
     in the conversation.
  6. Do not invent companies, periods, metrics or values.
  7. If the current question is already standalone,
     preserve its meaning without unnecessary changes.
  8. Return only the rewritten standalone question.
  """

  user_prompt = f"""
  Conversation history:
  {chat_history}

  Current question:
  {question}

  Rewrite the current question into a standalone question.  
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      },
    ],
    response_format=ContextualizedQuestion,
    temperature=0,
  )

  result = completion.choices[0].message.parsed

  if result is None:
    raise ValueError(
      "contextualize_question failed to generate "
      "ContextualizedQuestion"
    )

  return {
    "standalone_question": result.standalone_question,
  }

def intent_router(state: FinancialResearchState) -> dict:
  """
  判斷使用者問題屬於哪一種金融研究意圖。
  
  Node Input:
    FinancialResearchState
    
  Node output:
    更新 intent / companies / periods / router_confidence
  """

  question = state["question"]

  

  system_prompt = """
  你是一個金融研究問題路由器。
  請判斷使用者問題屬於以下哪一種 intent:
  
  1. single_company
  單一公司財報、營收、獲利、展望等問題。
  
  2. peer_comparison
  比較兩家或多家公司。
  
  3. industry_research
  產業研究、上下游研究、供應鏈研究、產業公司群研究
  
  4. general_financial_question
  不針對特定公司的通用金融問題。
  
  companies:
  找出問題中涉及的公司股票代號。
  如果無法判斷股票代號，可以先保留使用者使用的公司名稱
  
  periods:
  找出問題中的年度或季度，例如:
  2025Q4
  2025
  2026Q1
  
  如果沒有期間則回傳空陣列。
  
  confidence:
  0 到 1 之間，表示你對 intent 判斷的信心。
  """
  response = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": question
      }
    ],
    response_format=RouterResult,
    temperature=0,
  )

  router_result = response.choices[0].message.parsed

  if router_result is None:
    raise ValueError("intent_router 無法取得 structured output")

  return {
    "intent": router_result.intent,
    "companies": router_result.companies,
    "periods": router_result.periods,
    "router_confidence": router_result.confidence,
  }

class ResearchTask(BaseModel):
  """
  小問題(工作項目)
  """
  company: str | None
  period: str | None
  topic: str
  query: str

class ResearchPlan(BaseModel):
  """
  複雜問題,拆成多個可以交給 RAG 執行的小問題
  """
  research_goal: str
  tasks: list[ResearchTask]

class Evidence(BaseModel):
  """
  單一 ResearchTask 執行後取得的研究證據。
  """
  task_id: str
  company: str | None
  period: str | None
  query: str
  answer: str

  retrieved_contexts: list[str] # 實際送進 LLM 的文字內容，後續做 RAGAS、faithfulness 或 evidence checking 很有用

  """
  向量資料庫檢索結果的原始 metadata，例如 ticker、year、quarter 等,
  backend / debug / evaluation 用
  """
  metadata: list[dict] 

  """
  # 從 metadata 整理後要給 UI 或最終回答顯示的引用來源，例如檔名、公司、季度、文件標題
  Chainlit / report_writer / citation UI 用
  """
  sources: list[dict] 

class EvidenceCheckResult(BaseModel):
  """
  LLM 對 Evidence 的語意檢查結果。

  注意:
  failure_type 與 next_action 不由 LLM 決定，
  而是由 evidence_checker() 使用 Python 規則決定。

  """
  sufficient: bool
  missing_topics: list[str]
  weak_evidence: list[str]

class RegeneratedAnswer(BaseModel):
  """
  根據既有 retrieved_contexts 重新產生的回答
  """
  answer: str

class ReportResult(BaseModel):
  """
  report_writer 產生的最終研究回答
  """
  final_answer: str

class ContextualizedQuestion(BaseModel):
  standalone_question: str

def research_planner(state):
  """
  根據 intent_router 的結果，
  將研究問題拆成可以交給 RAG executor 執行的 ResearchTask。
  """

  question = state["question"]
  intent = state["intent"]
  companies = state["companies"]
  periods = state["periods"]

  system_prompt = """
  You are a research planner for a financial-report RAG system.

  Your job is to convert a user's financial research question
  into small, independent retrieval tasks that can be
  executed by a RAG system.
  
  The "topic" field must be a short financial topic or metric,
  not an instruction or sentence.
  Examples:
  "營收"
  "毛利率"
  "營業利益"
  "EPS"
  "資本支出"

  Rules:

  1. Each task must represent ONE retrieval objective.
  2. Each task should normally contain only ONE company.
  3. Each task should normally contain only ONE reporting period.
  4. The query must be standalone and understandable without conversation history.
  5. Only create tasks that can be answered from company financial reports.
  6. Do not answer the user's question.
  7. Do not invent companies, reporting periods, or financial facts.
  8. Use only companies and periods supplied by the router when they are available.
  9. For company comparisons, retrieve evidence for each company separately.
  10. Do not create a separate comparison task. Comparison will be performed later by the report writer.
  11. Keep the plan as small as possible while still answering the research goal.
  """

  user_prompt = f"""
  Original question:
  {question}

  Router result:

  intent:
  {intent}

  companies:
  {companies}

  periods:
  {periods}

  Create the research plan.
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      }
    ],
    response_format=ResearchPlan,
    temperature=0,
  )

  plan = completion.choices[0].message.parsed

  if plan is None:
    raise ValueError("research_planner failed to generate ResearchPlan")

  research_plan = []

  for index, task in enumerate(plan.tasks, start=1):

    """
    Notice: Structured Output 階段，暫時驗證 LLM 輸出
    等進到 LangGraph State 後再改成 Python dict
    """
    task_dict = task.model_dump()
    task_dict["task_id"] = f"task_{index}"
    research_plan.append(task_dict)

  return {
    "research_plan": research_plan,
    "current_task": 0,
  }

def build_sources(metadata_list: list[dict]) -> list[dict]:
  return [
    {
      "ticker": metadata.get("ticker"),
      "market": metadata.get("market"),
      "year": metadata.get("year"),
      "quarter": metadata.get("quarter"),
      "chunk_index": metadata.get("chunk_index"),
    }
    for metadata in metadata_list
  ]

def rag_executor(state: FinancialResearchState):
  """
  依序執行 research_planner 產生的 ResearchTask，
  並將每個任務的 RAG 結果整理成 Evidence。
  """

  evidence: list[dict] = []

  for task in state["research_plan"]:

    answer, retrieved_contexts, retrieved_metadata = rag_service.rag_task(
      user_query=task["query"],
      top_k=5,
    )

    # _rag_core() 回傳 dict[str, list[str]]
    # Evidence 使用 list[str]，所以要攤平成單一 list
    context_list = [
      context
      for contexts in retrieved_contexts.values()
      for context in contexts
    ]

    # _rag_core() 回傳 dict[str, list[str]]
    # Evidence 使用 list[str]，所以要攤平成單一 list
    metadata_list = [
      metadata
      for metadatas in retrieved_metadata.values()
      for metadata in metadatas
    ]

    # Notice: MVP 階段直接從 metadata 建立 sources
    # 後續 Chainlit UI 可以再做專門的 source formatter
    sources = build_sources(metadata_list)

    task_evidence = Evidence(
      task_id=task["task_id"],
      company=task.get("company"),
      period=task.get("period"),
      query=task.get("query"),
      answer=answer,
      retrieved_contexts=context_list,
      metadata=metadata_list,
      sources=sources
    )

    evidence.append(
      task_evidence.model_dump()
    )

  return {
    "evidence": evidence,
    "current_task": len(state["research_plan"])
  }

def retrieve_again(state: FinancialResearchState) -> dict:
  """
  當 evidence_checker 判斷目前 Evidence
  缺失或品質不足時，重新執行 retrieval。

  第一版策略:
  - 不修改 research_plan
  - 不修改 query
  - 增加 top_k
  - 使用新的 retrieval 結果取代原 evidence
  """

  evidence: list[dict] = []

  retrieval_count = state.get(
    "retrieval_count",
    0,
  )

  """
  第一次 retry 使用 top_k = 10
  第二次 retry 使用 top_k = 15
  """
  top_k = 10 + retrieval_count * 5

  for task in state["research_plan"]:

    answer, retrieved_contexts, retrieved_metadata = rag_service.rag_task(
      user_query=task["query"],
      top_k=top_k,
    )

    context_list = [
      context
      for contexts in retrieved_contexts.values()
      for context in contexts
    ]

    metadata_list = [
      metadata
      for metadatas in retrieved_metadata.values()
      for metadata in metadatas
    ]

    sources = build_sources(metadata_list)

    task_evidence = Evidence(
      task_id=task["task_id"],
      company=task.get("company"),
      period=task.get("period"),
      query=task.get("query"),
      answer=answer,
      retrieved_contexts=context_list,
      metadata=metadata_list,
      sources=sources,
    )

    evidence.append(
      task_evidence.model_dump(),
    )

  return {
    "evidence": evidence,
    "retrieval_count": retrieval_count + 1,
  }

def check_financial_unit_consistency(
  answer: str,
  retrieved_contexts: list[str],
) -> list[str]:
  """
  檢查 answer 與 retrieved contexts 中的財務數值單位是否明顯不一致

  第一版只處理:
  - billion
  - million
  - 億
  - 百萬

  統一轉換成 million 後比較。
  """
  issues = []

  context_text = "\n".join(retrieved_contexts)

  # -------------
  # 解析 answer
  # -------------

  answer_patterns = [
    # 例如: 1,046.09 億
    (
      rf"({FINANCIAL_NUMBER_PATTERN})\s*億",
      100.0, # 1 億 = 100 million
      "億",
    ),

    # 例如: 150, 188 百萬
    (
      rf"({FINANCIAL_NUMBER_PATTERN})\s*百萬",
      1.0,
      "百萬"
    ),

    # 例如: 1,046.09 billion
    (
      rf"({FINANCIAL_NUMBER_PATTERN})\s*billion",
      1000.0,
      "billion",
    ),

    #3 例如: 150,188 million
    (
      rf"({FINANCIAL_NUMBER_PATTERN})\s*million",
      1.0,
      "million",
    ),
  ]

  answer_values = []

  for pattern, multiplier, unit in answer_patterns:
    matches = re.findall(
      pattern,
      answer,
      flags=re.IGNORECASE,
    )

    for match in matches:

      cleaned_number = match.replace(",", "").strip()

      if not cleaned_number:
        continue

      value = float(cleaned_number)

      answer_values.append(
        {
          "raw_value": value,
          "unit": unit,
          "normalized": value * multiplier,
        }
      )


  # ----------------------------
  # 解析 retrieved contexts
  # ----------------------------

  context_values = []

  # 特別處理:
  # (In NT$ billions)
  # Net Revenue | 1,046.09

  if re.search(
    r"NT\$\s*billions?",
    context_text,
    flags=re.IGNORECASE,
  ):
    numbers = re.findall(
      FINANCIAL_NUMBER_PATTERN,
      context_text,
    )

    for number in numbers:
      cleaned_number = number.replace(",", "").strip()

      if not cleaned_number:
        continue

      value = float(cleaned_number)

      context_values.append(
        {
          "raw_value": value,
          "unit": "billion",
          "normalized": value * 1000.0,
        }
      )

  if re.search(
    r"millions?\s+of\s+New\s+Taiwan\s+dollars",
    context_text,
    flags=re.IGNORECASE,
  ):
    numbers = re.findall(
      FINANCIAL_NUMBER_PATTERN,
      context_text,
    )

    for number in numbers:
      cleaned_number = number.replace(",", "").strip()
      if not cleaned_number:
        continue

      value = float(cleaned_number)

      context_values.append(
        {
          "raw_value": value,
          "unit": "million",
          "normalized": value,
        }
      )

  # ---------------------------
  # 第一版 consistency check
  # ---------------------------

  if not answer_values or not context_values:
    return issues

  for answer_value in answer_values:
    matched = False
    for context_value in context_values:

      # normalized 全部以 million 表示
      difference = abs(
        answer_value["normalized"]
        - context_value["normalized"]
      )

      tolerance = max(
        abs(context_value["normalized"]) * 0.001,
        0.01,
      )

      if difference <= tolerance:
        matched = True
        break

    if not matched:
      issues.append(
        (
          f"Answer financial value "
          f"{answer_value['raw_value']} {answer_value['unit']} "
          f"is not consistent with the retrieved evidence."
        )
      )
  return issues


def evidence_checker(state: FinancialResearchState) -> dict:
  """
  檢查 rag_executor 取得的 evidence，
  判斷目前資料是否足以完成原始研究問題。

  Node Input:
    question
    research_plan
    evidence

  Node Output:
    sufficient
    missing_information
    weak_evidence

  """

  question = state["question"]
  research_plan = state["research_plan"]
  evidence = state["evidence"]

  # 沒有任何 evidence 時，不需要浪費一次 LLM call
  if not evidence:
    return {
      "sufficient": False,
      "missing_information": ["No evidence was retrieved."],
      "weak_evidence": [],
      "unsupported_answer": [],
      "failure_type": "missing_evidence",
      "next_action": "retrieve_again"
    }

  # ---------------------------------
  # Layer 1: deterministic validation
  # ---------------------------------
  numeric_issues = []

  for item in evidence:
    issues = check_financial_unit_consistency(
      answer=item["answer"],
      retrieved_contexts=item["retrieved_contexts"]
    )

    numeric_issues.extend(issues)

  if numeric_issues:
    return {
      "sufficient": False,
      "missing_information": [],
      "weak_evidence": [],
      "unsupported_answer": numeric_issues,
      "failure_type": "answer_not_supported",
      "next_action": "regenerate_answer"
    }

  # ----------------------------------
  # Layer 2: LLM semantic validation
  # ----------------------------------

  system_prompt = """
  You are an evidence checker for a financial-report RAG system.

  Your job is NOT to answer the user's financial question.

  Your job is to determine whether the retrieved evidence is sufficient
  to answer the original research question.

  Evaluate ONLY the supplied research plan and retrieved evidence.
  Do not use outside knowledge.
  Do not invent financial facts.

  Definitions:
  - missing_topics:
    Information required by the research plan but completely absent
    from the retrieved evidence.

  - weak_evidence:
    Evidence exists for a required topic, but the evidence
    is incomplete, ambiguous, irrelevant, internally inconsistent, or insufficiently supported by the retrieved contexts.

  Important:
  A completely missing topic belongs ONLY in missing_topics.
  Do NOT also report the same issue in weak_evidence.

  Evaluation rules:

  1. Check whether every required research task has corresponding evidence.

  2. For company comparisons, every required company must have evidence.

  3. For multi-period questions, every required period must have evidence.

  4. An answer alone is not sufficient.
     The answer must be supported by relevant retrieved_contexts.
  
  5. If no evidence exists for a required task:
    - add the missing information to missing_topics
    - do NOT add it to weak_evidence
  
  6. If evidence exists but is incomplete, ambiguous,
     irrelevant, contradictory, or poorly supported:
    - add the issue to weak_evidence
    - do NOT classify it as missing_topics unless the requried
      information is actually absent

  7. Set sufficient=true only when all required research tasks
     are covered and the available evidence is adequate to
     answer the original question.

  8. missing_topics and weak_evidence should be concise and specific.

  9. Do not perform the final comparison.
      Do not write the final financial report.

  10. Compare the generated answer against retrieved_contexts.

  11. Treat numerical inconsistencies as weak evidence, including:
      - incorrect values
      - incorrect units
      - incorrect periods
      - incorrect company attribution
      - unsupported calculations or conversions

  12. If retrieved_contexts contain sufficient evidence but the generated answer
      misrepresents that evidence, classify the issue as weak_evidence.

  13. Do not classify an answer-evidence inconsistency as missing_topics
      when the required source information is already present. 
  """

  user_prompt = f"""
  Original question:
  {question}

  Research plan:
  {research_plan}

  Retrieved evidence:
  {evidence}

  Determine whether the retrieved evidence is sufficient to complete the original research question.
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt,
      },
      {
        "role": "user",
        "content": user_prompt,
      }
    ],
    response_format=EvidenceCheckResult,
    temperature=0,
  )

  check_result = completion.choices[0].message.parsed

  if check_result is None:
    raise ValueError(
      "evidence_checker failed to generate EvidenceCheckResult"
    )

  # ------------------------------
  # 根據 Evidence 檢查結果決定下一步
  # ------------------------------

  if check_result.missing_topics:
    failure_type = 'missing_evidence'
    next_action = "retrieve_again"

  elif check_result.weak_evidence:
    failure_type = "weak_evidence"
    next_action = "retrieve_again"

  elif check_result.sufficient:
    failure_type = "none"
    next_action = "proceed"

  else:

    # 防止 LLM 出現: 
    # sufficient = False, 
    # 但 missing_topics/ weak_evidence 都是空的
    failure_type = 'weak_evidence'
    next_action = "retrieve_again"


  return {
    "sufficient": check_result.sufficient,
    "missing_information": check_result.missing_topics,
    "weak_evidence": check_result.weak_evidence,
    "unsupported_answer": [],
    "failure_type": failure_type,
    "next_action": next_action
  }

def answer_regenerator(state: FinancialResearchState) -> dict:
  """
  對 answer_not_supported 的 evidence 重新產生回答

  注意:
  - 不重新執行 retrieval
  - 只使用既有 retrieved_contexts
  - 只重新產生 answer 與 retrieved evidence
    明顯不一致的 Evidence
  """

  evidence = state["evidence"]
  regenerated_evidence = []

  for item in evidence:

    # 再次確認這筆 Evidence 是否真的存在
    # answer / retrieved_contexts 的財務數值不一致
    issues = check_financial_unit_consistency(
      answer=item["answer"],
      retrieved_contexts=item["retrieved_contexts"],
    )

    # 這筆 Evidence 沒問題，不需要浪費 LLM call
    if not issues:
      regenerated_evidence.append(item)
      continue

    system_prompt = """
    You are an answer regeneration component for a financial-report RAG system.

    Your task is to regenerate the answer using ONLY
    the supplied retrieved contexts.

    Rules:

    1. Do not use outside knowledge.
    2. Do not invent financial facts.
    3. Do not perform a new retrieval.
    4. Every financial number in the answer must be
       directly supported by the retrieved contexts.
    5. Preserve the EXACT financial unit used by the source.
    6. Do NOT translate or convert financial units unless explicitly required.
    7. If the source says "billion", keep "billion".
    8. If the source says "million", keep "million".
    9. Never change "billion" into "億" without mathematically
       converting the numerical value.
    10. Make sure the company and reporting period
        match the supplied query and evidence.
    11. Keep the answer concise.
    12. If the retrieved contexts do not contain enough
        information to answer the query, explicitly state
        that the available evidence is insufficient. 
    """

    user_prompt = f"""
    Query:
    {item["query"]}

    Previous answer:
    {item["answer"]}

    Detected consistency issues:
    {issues}

    Retrieved contexts:
    {item["retrieved_contexts"]}

    The previous answer FAILED validation.

    Correct every detected consistency issue.

    When a source explicitly states a financial unit,
    copy that unit exactly instead of translating it.

    Regenerate the answer so that it is faithfully
    supported by the retrieved contexts.
    """

    completion = client.chat.completions.parse(
      model=MODEL_NAME,
      messages=[
        {
          "role": "system",
          "content": system_prompt,
        },
        {
          "role": "user",
          "content": user_prompt
        }
      ],
      response_format=RegeneratedAnswer,
      temperature=0,
    )

    result = completion.choices[0].message.parsed

    if result is None:
      raise ValueError(
        "answer_regenerator failed to generate RegeneratedAnswer"
      )

    """
    不改 retrieved_contexts / metadata /sources , 
    只替換 answer
    """
    updated_item = item.copy()
    updated_item["answer"] = result.answer

    regenerated_evidence.append(updated_item)

  return {
    "evidence": regenerated_evidence,
    "regeneration_count": state.get(
      "regeneration_count",
      0,
    ) + 1,
    "unsupported_answer": []
  }


def failure_report_writer(state: FinancialResearchState) -> dict:

  failure_type = state["failure_type"]

  if failure_type == 'answer_not_supported':
    final_answer = (
      "目前產生的財務數值無法通過證據一致性驗證"
      "且已達到自動修正次數上限，因此本次不提供未經驗證的財務結論。"
    )

  elif failure_type == 'missing_evidence':
    final_answer = (
      "目前找到的資料不足以完整回答這個問題，"
      "且已達到重新檢索次數上限"
    )

  elif failure_type == 'weak_evidence':
    final_answer = (
      "目前取得的證據不足以可靠支持完整結論，"
      "且已達到重新檢索次數上限。"
    )

  else: 
    final_answer = (
      "本次研究流程無法產生通過驗證的最終回答。"
    )

  return {
    "final_answer": final_answer,
  }

def report_writer(state: FinancialResearchState) -> dict:
  """
  將已通過 evidence_checker 驗證的 Evidence
  整合成最終回答
  
  注意:
  - 不執行 retrieval
  - 不修改 evidence
  - 不使用外部知識
  - 只使用已驗證 Evidence
  """

  question = state["question"]
  evidence = state["evidence"]

  system_prompt = """
  You are a financial research report writer for a RAG system.

  Your job is to answer the user's original question
  using ONLY the supplied validated evidence.

  Rules:

  1. Use only the supplied evidence.
  2. Do not use outside knowledge.
  3. Do not invent financial facts.
  4. Every financial number must be supported by the evidence.
  5. Preserve the financial units used by the evidence whenever possible.
  6. Make sure company names and reporting periods are correct.
  7. If the question compares multiple companies, perform the
     comparison using the evidence for each company.
  8. For comparison questions, do not merely list the values.
     Explicitly state which company has the higher value.
  9. When necessary for comparison, convert financial values
     to the same unit before calculating differences or ratios.
  10. Any unit conversion or calculation must be mathematically correct.
  11. When comparing numerical values, include the difference
      or ratio when it helps answer the question.
  12. Use financial terminology carefully.
      Do not translate "Net Revenue" as "淨收入".
      Use "營收" or "淨營收" instead.
  13. Do not claim information that is not present in the evidence.
  14. Keep the answer concise and suitable for a financial research response.
  15. Answer the original question directly.
  """
  # ==========
  # Notice: 第11 和12點是 MVP/Demo 階段暫時解，之後再用更好方式解
  # ==========

  user_prompt = f"""
  Original question:
  {question}

  Validated evidence:
  {evidence}
  
  Write the final answer.
  """

  completion = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": user_prompt
      }
    ],
    response_format=ReportResult,
    temperature=0,
  )

  result = completion.choices[0].message.parsed

  if result is None:
    raise ValueError(
      "report_writer failed to generate ReportResult"
    )

  return {
    "final_answer": result.final_answer,
  }