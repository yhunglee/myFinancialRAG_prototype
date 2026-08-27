from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

@dataclass
class RAGEvent:
  type: Literal[
    "reasoning",
    "answer",
  ]

  content: str