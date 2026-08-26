from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

from openai import AsyncOpenAI

import instructor
from ragas.llms import InstructorLLM

from ragas.metrics.collections import (
  Faithfulness,
  FactualCorrectness,
  ContextRecall,
)
import math



# ========================================
# Context
# ========================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ragas_intermediate_dataset.json"

OUTPUT_ALL = BASE_DIR / "ragas_results_all.csv"
OUTPUT_REFERENCE = BASE_DIR / "ragas_results_with_reference.csv"

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

# TODO: add model ID
JUDGE_MODEL = "qwen3.5-9b"

# =======================================
# Utility
# =======================================

def remove_think_block(text: str) -> str:
  """
  移除模型輸出中的 <think>...</think>
  避免把 reasoning trace 當成正式答案交給 RAGAS 評分
  """

  if not text:
    return ""

  cleaned = re.sub(
    r"<think>.*?</think>",
    "",
    text,
    flags=re.DOTALL | re.IGNORECASE,
  )

  return cleaned.strip()

def load_intermediate_dataset() -> list[dict]:
  """
  讀取 RAG pipeline 已經產生好的中間資料。
  """

  with INPUT_FILE.open(
    "r",
    encoding="utf-8",
  ) as f:
    data = json.load(f)

  print(f"Loaded {len(data)} samples.")

  return data

def normalize_samples(data: list[dict]) -> list[dict]:
  """
  只剩下 RAGAS EvalutionDataset 需要的欄位，
  並清除 response 裡的 <think> block
  """

  samples = []

  for index, item in enumerate(data):

    cleaned_response = remove_think_block(
      item["response"]
    )

    if not cleaned_response:
      print(
        f"WARNING sample {index}: "
        f"response became empty after cleaning"
      )
      print(
        "question:",
        item["user_input"],
      )

    sample = {
      "user_input": item["user_input"],
      "retrieved_contexts":
        item["retrieved_contexts"],

      "response":
        cleaned_response,

      "reference":
        item.get(
          "reference",
          "",
        ).strip(),
    }

    samples.append(sample)

  return samples


# ==========================================
# Judge LLM
# ==========================================

def create_judge_llm():
  client = AsyncOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",

    # Local model 有時推論較慢
    timeout=180.0,
    max_retries=2,
  )

  # ----------------------------------------
  # 2. 用 Instructor 包裝
  #
  # 關鍵：
  # LM Studio 支援 json_schema，
  # 不接受 json_object。
  # ----------------------------------------
  instructor_client = instructor.from_openai(
    client,
    mode=instructor.Mode.JSON_SCHEMA,
  )

  # ----------------------------------------
  # 3. 建立 RAGAS InstructorLLM
  #
  # 不再呼叫 llm_factory()，
  # 避免 RAGAS 0.4.3 再次使用預設 Mode.JSON
  # ----------------------------------------
  judge_llm = InstructorLLM(
    client=instructor_client,
    model=JUDGE_MODEL,
    provider="openai",

    # RAGAS 預設只有 1024，
    # 增加 Judge structured output 空間
    # 注意：
    # Qwen Judge 的 thinking 模式目前由 LM Studio
    # Developer -> Inference 設定控制。
    # 必須停用 thinking，否則 RAGAS 的 structured output
    # 容易因 reasoning 消耗大量 output tokens 而被截斷。
    max_tokens=4096,
  )

  return judge_llm


# ==========================================
# Evaluation 1
# Faithfulness
# ==========================================

def evaluate_faithfulness(
    samples: list[dict],
    judge_llm,
):

    print("\n========================================")
    print("Evaluating Faithfulness")
    print("========================================")

    metric = Faithfulness(
        llm=judge_llm
    )

    rows = []

    total = len(samples)

    for index, sample in enumerate(samples, start=1):

        print(
            f"\n[{index}/{total}] "
            f"Faithfulness: {sample['user_input']}"
        )

        # 每一筆 sample 都先初始化評分與錯誤狀態
        score = None
        error = None

        try:
            result = metric.score(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            )

            score = float(result.value)

            # 檢查 NAN
            if math.isnan(score):

              score = None
              error = "Faithfulness returned NaN"

              print(
                "    WARNING: "
                "Faithfulness returned NaN"
              )

            else:
              print(
                f"    score = {score:.4f}"
              )

        except Exception as e:

            print(
                f"    ERROR: "
                f"{type(e).__name__}: {e}"
            )

            score = None
            error = (
                f"{type(e).__name__}: {e}"
            )

        # ----------------------------------------
        # Save result
        # ----------------------------------------
        row = {
            "user_input":
                sample["user_input"],

            "response":
                sample["response"],

            "reference":
                sample.get(
                    "reference",
                    "",
                ),

            "faithfulness":
                score,

            "faithfulness_error":
                error,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_ALL,
        index=False,
        encoding="utf-8-sig",
    )

    return df


# =========================================
# Evaluation 2
# Reference-dependent metrics
# =========================================

def evaluate_reference_metrics(
    samples: list[dict],
    judge_llm,
):

  reference_samples = [
      sample
      for sample in samples
      if sample.get("reference", "").strip()
  ]

  print(
      f"\nSamples with reference: "
      f"{len(reference_samples)} / {len(samples)}"
  )

  if not reference_samples:
      print(
          "No samples contain reference answers."
      )
      return None

  print("\n========================================")
  print("Evaluating Reference Metrics")
  print("========================================")

  factual_metric = FactualCorrectness(
      llm=judge_llm
  )

  context_recall_metric = ContextRecall(
      llm=judge_llm
  )

  rows = []

  total = len(reference_samples)

  for index, sample in enumerate(
      reference_samples,
      start=1,
  ):

    print(
        f"\n[{index}/{total}] "
        f"{sample['user_input']}"
    )

    # ------------------------------------
    # Factual Correctness
    # ------------------------------------

    factual_score = None
    factual_error = None

    try:

      factual_result = factual_metric.score(
          response=sample["response"],
          reference=sample["reference"],
      )

      factual_score = float(
          factual_result.value
      )

      print(
          f"    factual_correctness "
          f"= {factual_score:.4f}"
      )

    except Exception as e:

      print(
          f"    FactualCorrectness ERROR: "
          f"{type(e).__name__}: {e}"
      )

      factual_score = None
      factual_error = (
          f"{type(e).__name__}: {e}"
      )

    # ------------------------------------
    # Context Recall
    # ------------------------------------

    recall_score = None
    recall_error = None

    try:

      recall_result = (
          context_recall_metric.score(
              user_input=sample["user_input"],
              retrieved_contexts=(
                  sample["retrieved_contexts"]
              ),
              reference=sample["reference"],
          )
      )

      recall_score = float(
          recall_result.value
      )

      print(
          f"    context_recall "
          f"= {recall_score:.4f}"
      )

    except Exception as e:

      print(
          f"    ContextRecall ERROR: "
          f"{type(e).__name__}: {e}"
      )

      recall_score = None
      recall_error = (
          f"{type(e).__name__}: {e}"
      )

    # ------------------------------------
    # Save row
    # ------------------------------------

    row = {
      "user_input": sample["user_input"],
      "response": sample["response"],
      "reference": sample["reference"],

      "factual_correctness":
          factual_score,

      "factual_correctness_error":
          factual_error,

      "context_recall":
          recall_score,

      "context_recall_error":
          recall_error,
    }

    rows.append(row)

  df = pd.DataFrame(rows)

  df.to_csv(
      OUTPUT_REFERENCE,
      index=False,
      encoding="utf-8-sig",
  )

  return df

# ===========================================
# Summary
# ===========================================

def print_summary(
    faithfulness_df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
):

  print("\n========================================")
  print("RAGAS Evaluation Summary")
  print("========================================")

  if (
    faithfulness_df is not None
    and "faithfulness"
    in faithfulness_df.columns
  ):

    valid_scores = (
      faithfulness_df["faithfulness"]
      .dropna()
    )

    if valid_scores.empty:
      print(
        f"{'faithfulness':25s}: "
        "NO VALID SCORES"
      )
    else:
      print(
        f"{'faithfulness':25s}: "
        f"{valid_scores.mean():.4f}"
      )

  if reference_df is not None:

    for column in [
      "factual_correctness",
      "context_recall",
    ]:

      if column in reference_df.columns:

        valid_scores = (
          reference_df[column]
          .dropna()
        )

        if not valid_scores.empty:

          print(
              f"{column:25s}: "
              f"{valid_scores.mean():.4f}"
          )



# ==================================
# Main
# ==================================

def main():

  raw_data = load_intermediate_dataset()

  samples = normalize_samples(raw_data)

  if len(samples) != len(raw_data):
    raise ValueError(
        "Dataset normalization changed sample count: "
        f"{len(raw_data)} -> {len(samples)}"
    )

  judge_llm = create_judge_llm()

  faithfulness_df = evaluate_faithfulness(
    samples=samples,
    judge_llm=judge_llm,
  )

  reference_df = evaluate_reference_metrics(
    samples=samples,
    judge_llm=judge_llm,
  )

  print_summary(
    faithfulness_df,
    reference_df,
  )

  print("\nSaved:")
  print(OUTPUT_ALL)
  print(OUTPUT_REFERENCE)

if __name__ == "__main__":
  main()