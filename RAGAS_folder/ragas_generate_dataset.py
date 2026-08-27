import json

import sys, os
# 1. 自動將父資料夾加入搜尋路徑（解決 myrag_module 的導入問題）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myrag_module import FinancialRAGService

# 2. 自動修正 JSON 檔案的讀取路徑（解決 FileNotFoundError）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(CURRENT_DIR, "base_dataset.json")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "ragas_intermediate_dataset.json")




def flatten_contexts(retrieved_contexts: dict) -> list[str]:
  contexts = []

  for company, chunks in retrieved_contexts.items():
      for chunk in chunks:
          contexts.append(chunk)
  return contexts

def flatten_metadata(retrieved_metadata: dict) -> list[dict]:
  metadata_list = []

  for company, metadatas in retrieved_metadata.items():
      for metadata in metadatas:
          metadata_list.append(metadata)

  return metadata_list


with open(INPUT_FILE, "r", encoding="utf-8") as f:
  dataset = json.load(f)

results = []
  
rag = FinancialRAGService()

for index,item in enumerate(dataset, start=1):
  question = item["user_input"]
  print("=" * 60)
  print(f"[{index}/{len(dataset)}]")
  print("問題:", question)

  # 每一題獨立測試
  rag.clean_history()

  answer, retrieved_contexts, retrieved_metadata = rag.rag_chat(
    question,
    top_k=5
  )


  # 轉成 evaluation 需要的格式
  contexts = flatten_contexts(retrieved_contexts)
  metadata = flatten_metadata(retrieved_metadata)

  result = {
     "user_input": question,
     "retrieved_contexts": contexts,
     "retrieved_metadata": metadata,
     "response": answer,
     "reference": item["reference"]
  }

  results.append(result)


# 寫入新的 JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
   json.dump(
      results,
      f,
      ensure_ascii=False,
      indent=2
   )

print()
print("RAGAS 中間檔完成:")
print(OUTPUT_FILE)