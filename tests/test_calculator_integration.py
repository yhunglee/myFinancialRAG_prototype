from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

# 允許從專案根目錄匯入 agent_graph
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_graph import build_agent_graph


EXPECTED_DIFFERENCE = Decimal("895902")


def create_initial_state(
    question: str,
    chat_history: list[dict] | None = None,
    previous_validated_evidence: list | None = None,
) -> dict:
    """
    建立與目前 Chainlit app.py 相同的初始 State。
    測試不需要啟動 Chainlit UI。
    """
    return {
        "question": question,
        "standalone_question": "",
        "chat_history": chat_history or [],
        "intent": "",
        "companies": [],
        "periods": [],
        "router_confidence": 0.0,
        "research_plan": [],
        "current_task": 0,
        "previous_validated_evidence": (
            previous_validated_evidence or []
        ),
        "reuse_evidence": False,
        "evidence": [],
        "task_results": [],
        "sufficient": False,
        "missing_information": [],
        "weak_evidence": [],
        "unsupported_answer": [],
        "failure_type": "none",
        "next_action": "proceed",
        "regeneration_count": 0,
        "retrieval_count": 0,
        "calculation_required": False,
        "calculation_results": [],
        "final_answer": "",
    }


async def invoke_graph(
    question: str,
    chat_history: list[dict] | None = None,
    previous_validated_evidence: list | None = None,
) -> dict:
    graph = build_agent_graph()

    return await graph.ainvoke(
        create_initial_state(
            question,
            chat_history,
            previous_validated_evidence,
        )
    )


def verify_result(
    result: dict,
    expected_calculation: bool,
) -> None:
    """
    驗證是否正確使用 Calculator，
    並檢查實際執行的計算結果。
    """

    print("\n========== TEST DIAGNOSTICS ==========")
    print("Question:", result.get("standalone_question"))
    print("Sufficient:", result.get("sufficient"))
    print("Failure type:", result.get("failure_type"))
    print("Missing information:", result.get("missing_information"))
    print("Weak evidence:", result.get("weak_evidence"))
    print("Unsupported answer:", result.get("unsupported_answer"))
    print("Next action:", result.get("next_action"))
    print("Calculation required:", result.get("calculation_required"))
    print("Calculation results:", result.get("calculation_results"))
    print("Final answer:", result.get("final_answer"))
    print("======================================\n")

    print("\n========== EVIDENCE DETAILS ==========")

    for item in result.get("evidence", []):
        print("Task ID:", item.get("task_id"))
        print("Company:", item.get("company"))
        print("Period:", item.get("period"))
        print("Topic:", item.get("topic"))
        print("Answer:", item.get("answer"))
        print(
            "Retrieved contexts:",
            item.get("retrieved_contexts")
        )
        print("--------------------------------------")

    print("Research Plan:", result.get("research_plan"))
    print("Task Results:", result.get("task_results"))

    print("======================================\n")

    assert result["sufficient"] is True, (
        f"Evidence validation failed: "
        f"{result.get('failure_type')}"
    )

    assert bool(result["final_answer"].strip()), (
        "Report Writer did not produce an answer."
    )

    assert (
        result["calculation_required"]
        is expected_calculation
    ), (
        "Incorrect Calculator tool selection. "
        f"Expected: {expected_calculation}, "
        f"Actual: {result['calculation_required']}"
    )

    calculation_results = result["calculation_results"]

    if not expected_calculation:
        assert calculation_results == [], (
            "Calculator executed unexpectedly."
        )
        return

    assert len(calculation_results) == 1, (
        "Expected exactly one calculation result."
    )

    calculation = calculation_results[0]

    assert calculation["operation"] == "difference"
    assert calculation["unit"] == "million"

    assert (
        Decimal(str(calculation["result"]))
        == EXPECTED_DIFFERENCE
    ), (
        "Incorrect revenue difference: "
        f"{calculation['result']}"
    )

    # 同時檢查單位轉換後的兩個運算元
    actual_operands = {
        Decimal(str(calculation["operand_a"]["normalized_value"])),
        Decimal(str(calculation["operand_b"]["normalized_value"])),
    }

    assert actual_operands == {
        Decimal("1046090"),
        Decimal("150188"),
    }, "Incorrect normalized financial operands."


@pytest.mark.parametrize(
    "case_id, question, expected_calculation",
    [
        (
            "TC01",
            "比較台積電和聯發科 2025 年第四季營收",
            False,
        ),
        (
            "TC02",
            "台積電和聯發科在 2025 年第四季的營收相差多少？",
            True,
        ),
        (
            "TC04",
            "台積電和聯發科在 2025 年第四季，誰的營收比較高？",
            False,
        ),
        (
            "TC05",
            "台積電 2025 年第四季的營收是多少？",
            False,
        ),
        (
            "TC06",
            "台積電在 2025 年第四季的營收比聯發科高多少？",
            True,
        ),
    ],
)
def test_calculator_tool_selection(
    case_id: str,
    question: str,
    expected_calculation: bool,
):
    """
    每個案例均使用獨立對話狀態。
    """
    print(f"\n[{case_id}] {question}")

    result = asyncio.run(invoke_graph(question))

    verify_result(result, expected_calculation)

    print(f"[{case_id}] PASS")


def test_tc03_multi_turn_calculation():
    """
    TC03: 先比較營收，再追問差額。
    模擬 Chainlit 中同一個 Session 的對話。
    """
    first_question = (
        "比較台積電和聯發科 2025 年第四季營收"
    )

    first_result = asyncio.run(
        invoke_graph(first_question)
    )

    # 先確認第一輪研究成功
    verify_result(first_result, False)

    history = [
        {
            "role": "user",
            "content": first_question,
        },
        {
            "role": "assistant",
            "content": first_result["final_answer"],
        },
    ]

    follow_up = "那兩家公司的營收差多少？"

    print(f"\n[TC03] {follow_up}")

    second_result = asyncio.run(
        invoke_graph(
            question=follow_up,
            chat_history=history,
            previous_validated_evidence=(
                first_result["evidence"]
            ),
        )
    )

    # 確認追問被改寫為可獨立理解的問題
    standalone = second_result["standalone_question"]

    assert standalone.strip(), (
        "Standalone question is empty."
    )

    verify_result(second_result, True)

    print(f"[TC03] Standalone: {standalone}")
    print("[TC03] PASS")
