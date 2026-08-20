from __future__ import annotations

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 1. 載入檢索與向量模型
embedding_model = SentenceTransformer(
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="financial_reports")

# 2. 檢查資料庫狀態
total_count = collection.count()
print(f"📊 目前 ChromaDB 中的財報總區塊數 (Chunks): {total_count}")

if total_count == 0:
  print("❌ 資料庫內無資料，請先確認 Ingestion 腳本是否成功執行！")
  exit()

# 3. 連結 LM Studio 本機模型
llm_client = OpenAI(
  base_url="http://localhost:1234/v1",
  api_key="lm-studio" # 任意填值，因沒開認證
)

# 4. 定義測試問題
question = "台積電 2025 年第四季的合併營收是多少?"
print(f"\n[提問] {question}")

# 5. 執行向量檢索 (可搭配 where 進行 Metadata 精確過濾)
query_vector = embedding_model.encode([question], normalize_embeddings=True).tolist()

results = collection.query(
  query_embeddings=query_vector,
  n_results=3,
  where={"ticker": "2330"} # metadata 鎖定台積電
)

retrieved_docs = results["documents"][0] if results["documents"] else []
retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []

if not retrieved_docs:
    print("⚠️ 檢索未命中任何相關資料，請檢查 where 條件或關鍵字。")
    exit()

# 6. 組合帶有引用來源的 Context
context_blocks = []
for doc, meta in zip(retrieved_docs, retrieved_metadatas):
  context_blocks.append(f"[來源: {meta.get('ticker')} {meta.get('year')} {meta.get('quarter')} Chunk-{meta.get('chunk_index')}]\n{doc}")

context_text = "\n\n".join(context_blocks)

system_prompt = (
  "你是一名專業的台美股財務分析師。請嚴格依據以下提供的參考資料回答問題。"
  "必須清楚標示引用自哪一段資料。若資料未提及具體數字，請直說不知道，切勿自行猜測。"
)

user_prompt = f"""[參考資料]
{context_text}

[問題]
{question}
"""

# 7. 生成回答
response = llm_client.chat.completions.create(
  model="local-model", # 任意名稱
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ],
  temperature=0.1,
  max_tokens=2048,
)

choice = response.choices[0]
final_answer = choice.message.content

# 若使用推理模型，在 content 為空時印出思考內容供除錯
if not final_answer and hasattr(choice.message, "reasoning_content"):
    final_answer = f"[模型思考輸出截斷]\n{choice.message.reasoning_content}"

print("\n=== AI 財報分析師回答 ===")
print(final_answer)