from __future__ import annotations

from pprint import pprint

from myrag_module import FinancialRAGService

rag_service = FinancialRAGService(
  db_path="./chroma_db",
  collection_name="financial_reports",
)

query = "台積電 2025年第四季營收"

messages, retrieved_contexts, retrieved_metadata=  rag_service._rag_core(
  user_query=query,
  top_k=5,
)

print("\n=========================")
print("Retrieved Metadata")
print("----------------------------")

print(retrieved_metadata)

print("\n=========================")
print("Metadata Keys")
print("===========================")

all_keys = set()

for company_or_ticker, metadata_list in retrieved_metadata.items():

  print(f"\nGroup: {company_or_ticker}")

  for index, metadata in enumerate(metadata_list, start=1):

    print(f"\n  Metadata #{index}")
    pprint(metadata)

    all_keys.update(metadata.keys())


print("\n=====================")
print("All Metadata Keys")
print("=======================")

for key in sorted(all_keys):
  print(key)