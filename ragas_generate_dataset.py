import json
from myrag_module import FinancialRAGService

INPUT_FILE = "./RAGAS_folder/base_dataset.json"
OUTPUT_FILE = "./RAGAS_folder/ragas_intermediate_dataset.json"




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
  
  answer, retrieved_contexts, retrieved_metadata = rag.rag_chat(
    question,
    top_k=5
  )