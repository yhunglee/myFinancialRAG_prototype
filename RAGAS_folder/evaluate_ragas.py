from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

from openai import AsyncOpenAI

from ragas.llms import llm_factory

from ragas.metrics.collections import (
  Faithfulness,
  FactualCorrectness,
  ContextRecall,
)


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

    sample = {
      "user_input": item["user_input"],
      "retrieved_contexts": item["retrieved_contexts"],
      "response": remove_think_block(
        item["response"]
      ),
      "reference": item.get("reference", "")
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

  judge_llm = llm_factory(
    model=JUDGE_MODEL,
    client=client,
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

    try:

      result = metric.score(
          user_input=sample["user_input"],
          response=sample["response"],
          retrieved_contexts=sample["retrieved_contexts"],
      )

      score = float(result.value)

      reason = getattr(
          result,
          "reason",
          None,
      )

      print(
          f"    score = {score:.4f}"
      )

    except Exception as e:

      print(
          f"    ERROR: {type(e).__name__}: {e}"
      )

      score = None
      reason = str(e)

    row = {
      "user_input": sample["user_input"],
      "response": sample["response"],
      "reference": sample.get(
          "reference",
          "",
      ),
      "faithfulness": score,
      "faithfulness_reason": reason,
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

    try:

      factual_result = factual_metric.score(
          response=sample["response"],
          reference=sample["reference"],
      )

      factual_score = float(
          factual_result.value
      )

      factual_reason = getattr(
          factual_result,
          "reason",
          None,
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
      factual_reason = str(e)

    # ------------------------------------
    # Context Recall
    # ------------------------------------

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

      recall_reason = getattr(
          recall_result,
          "reason",
          None,
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
      recall_reason = str(e)

    # ------------------------------------
    # Save row
    # ------------------------------------

    row = {
      "user_input": sample["user_input"],
      "response": sample["response"],
      "reference": sample["reference"],

      "factual_correctness":
          factual_score,

      "factual_correctness_reason":
          factual_reason,

      "context_recall":
          recall_score,

      "context_recall_reason":
          recall_reason,
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

    score = (
      faithfulness_df["faithfulness"]
      .dropna()
      .mean()
    )

    print(
      f"{'faithfulness':25s}: "
      f"{score:.4f}"
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

  # debug mode. TODO: remove 
  samples = samples[:3]
  print(
    f"DEBUG mode: evaluating only "
    f"{len(samples)} samples."
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