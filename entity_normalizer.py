import re
from typing import Dict, List, Optional, TypedDict

class StockMetadata(TypedDict):
  canonical_ticker: str # 標準代號(例如: "2330.TW", "NVDA")
  formal_name: str # 正式全名(例如: "台灣積體電路製造", "NVIDIA Corporation")
  market: str # 市場("TW" 或 "US")
  aliases: List[str] # 別名、縮寫、俗稱

class StockEntityNormalizer:
  """
  台美股實體對齊與別名規範化模組
  功能:
  1. 將使用者輸入的俗稱(如:'GG', '發哥', '老黃') 對齊為標準代號(如: '2330.TW, 'NVDA')
  2. 從自然語言句子中提取包含的多家公司實體(支援比較查詢)
  3. 校驗與修復 LLM 輸出的非標準 Ticker
  """

  def __init__(self):
    # 1. 定義標準股票資料庫(可隨時擴充或改由 JSON/YAML 載入)
    self.stock_database: Dict[str, StockMetadata] = {
      # 台股清單
      "2330.TW": {
        "canonical_ticker": "2330.TW",
        "formal_name": "台積電",
        "market": "TW",
        "aliases": ['台積電', '台積', '2330', 'TSMC', 'TSM', 'GG', '護國神山']
      },
      "2454.TW": {
        "canonical_ticker": "2454.TW",
        "formal_name": "聯發科",
        "market": "TW",
        "aliases": ["聯發科", "發哥", "2454", "MediaTek", "MTK", "聯發科技"]
      },
      "2317.TW": {
          "canonical_ticker": "2317.TW",
          "formal_name": "鴻海",
          "market": "TW",
          "aliases": ["鴻海", "2317", "富士康", "Foxconn", "海公公"]
      },
      "2382.TW": {
          "canonical_ticker": "2382.TW",
          "formal_name": "廣達",
          "market": "TW",
          "aliases": ["廣達", "2382", "Quanta", "廣達電腦"]
      },

      # === 美股清單 ===
      "NVDA": {
          "canonical_ticker": "NVDA",
          "formal_name": "NVIDIA",
          "market": "US",
          "aliases": ["輝達", "英偉達", "NVDA", "NVIDIA", "老黃", "黃仁勳", "皮衣刀客", "綠廠"]
      },
      "AMD": {
          "canonical_ticker": "AMD",
          "formal_name": "Advanced Micro Devices",
          "market": "US",
          "aliases": ["超微", "AMD", "蘇媽", "超微半導體", "紅廠"]
      },
      "AAPL": {
          "canonical_ticker": "AAPL",
          "formal_name": "Apple Inc.",
          "market": "US",
          "aliases": ["蘋果", "蘋果公司", "AAPL", "Apple", "果子"]
      },
      "MSFT": {
          "canonical_ticker": "MSFT",
          "formal_name": "Microsoft",
          "market": "US",
          "aliases": ["微軟", "MSFT", "Microsoft", "軟微"]
      },
      "TSLA": {
          "canonical_ticker": "TSLA",
          "formal_name": "Tesla",
          "market": "US",
          "aliases": ["特斯拉", "TSLA", "Tesla", "馬斯克"]
      }
    }

    """
      2. 自動建立反向對照表 (Alias -> Canonical Ticker)
      例如: "gg" -> "2330.TW", "發哥" -> "2454.TW"
    """
    self.alias_to_ticker: Dict[str, str] = {}
    self._build_reverse_index()

  def _build_reverse_index(self):
    """建立不區分大小寫的反向對照表"""
    for ticker, meta in self.stock_database.items():
      # 先綁訂標準代號本身
      self.alias_to_ticker[ticker.lower()] = ticker
      self.alias_to_ticker[ticker.replace(".TW", "").lower()] = ticker

      # 綁定所有別名
      for alias in meta["aliases"]:
        self.alias_to_ticker[alias.lower()] = ticker

  def normalize(self, raw_entity: str) -> Optional[StockMetadata]:
    """
    單一實體規範化: 將任意字串轉為標準股票資訊
    例如輸入 "發哥" -> 回傳 2454.TW 的完整 Metadata
    """

    if not raw_entity:
      return None

    cleaned_text = raw_entity.strip().lower()
    canonical_ticker = self.alias_to_ticker.get(cleaned_text)

    if canonical_ticker:
      return self.stock_database[canonical_ticker]

    return None

  def extract_entities_from_text(self, text: str) -> List[StockMetadata]:
    """
    從自然語言句子中取得所有提到的股票實體(依長度優先比對，避免誤判)
    例如: "我想比較 GG 和 NVDA 還有發哥在 2023 的毛利"
    -> 回傳 [2330.TW, NVDA, 2454.TW] 的 Metadata 列表
    """
    matched_tickers = set()
    matched_metadata = []

    # 依照別名長度由長到短比對排序(例如先比對 "台積電"，再比對 "台積"，避免截斷")
    sorted_aliases = sorted(self.alias_to_ticker.keys(), key=len, reverse=True)

    text_lower = text.lower()
    for alias in sorted_aliases:
      # 使用邊界或包含判斷(中文直接包含，英文用詞邊界判斷)
      if alias.isascii():
        is_matched = re.search(
          rf'\b{re.escape(alias)}\b',
          text_lower
        )
      else:
        is_matched = alias in text_lower

      if is_matched:
        canonical_ticker = self.alias_to_ticker[alias]
        if canonical_ticker not in matched_tickers:
          matched_tickers.add(canonical_ticker)
          matched_metadata.append(self.stock_database[canonical_ticker])

    return matched_metadata