from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

from openai import AsyncOpenAI

from ragas import EvaluationDataset, evaluate
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
  print("\n==================================")
  print("Evaluatiing Faithfulness")
  print("====================================")

  dataset = EvaluationDataset.from_list(samples)

  result = evaluate(
    dataset=dataset,
    metrics=[
      Faithfulness(),
    ],
    llm=judge_llm,

    # Local LLM 先不要同時使用太多 request
    batch_size=1,

    raise_exceptions=False,
    show_progress=True,
  )

  df = result.to_pandas()

  df.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
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
    if sample["reference"]
  ]

  print(
    f"\nSamples with reference: "
    f"{len(reference_samples)} / {len(samples)}"
  )

  if not reference_samples:
    print("No samples contain reference answers.")
    return None

  dataset = EvaluationDataset.from_list(
    reference_samples
  )

  print("\n===================================")
  print("Evaluating reference-based metrics")
  print("====================================")

  result = evaluate(
    dataset=dataset,
    metrics=[
      FactualCorrectness(),
      ContextRecall(),
    ],
    llm=judge_llm,
    batch_size=1,
    raise_exceptions=False,
    show_progress=True,
  )

  df = result.to_pandas()

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
  print("\n======================================")
  print("RAGAS Evaluation Summary")
  print("=======================================")

  metric_columns = [
    "faithfulness",
    "factual_correctness",
    "context_recall",
  ]

  for df in [faithfulness_df, reference_df]:

    if df is None:
      continue

    for column in metric_columns:
      if column in df.columns:

        score = df[column].mean()

        print(
          f"{column:25s}: "
          f"{score:.4f}"
        )



# ==================================
# Main
# ==================================

def main():

  raw_data = load_intermediate_dataset()

  samples = normalize_samples(raw_data)

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