
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_node import calculator


# 每題重複執行次數
REPEAT_COUNT = 10

EXPECTED_DIFFERENCE = Decimal("895902")


# 固定 Evidence，不經過 RAG Retrieval
FIXED_EVIDENCE = [
    {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "topic": "營收",
        "answer": (
            "台積電在 2025Q4 的營收（Net Revenue）"
            "為 NT$1,046.09 billion。"
        ),
        "retrieved_contexts": [
            (
                "TSMC Fourth Quarter 2025. "
                "(In NT$ billions) | 4Q25 | 3Q25\n"
                "Net Revenue | 1,046.09 | 989.92"
            )
        ],
    },
    {
        "task_id": "task_2",
        "company": "聯發科",
        "period": "2025Q4",
        "topic": "營收",
        "answer": (
            "聯發科在 2025Q4 的營收（Net Sales）"
            "為 150,188 Millions of New Taiwan dollars。"
        ),
        "retrieved_contexts": [
            (
                "MediaTek Fourth Quarter 2025. "
                "Unit: Millions of New Taiwan dollars.\n"
                "Net Sales | 150,188"
            )
        ],
    },
]


TEST_CASES = [
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
]


def verify_calculator_output(
    result: dict,
    expected_required: bool,
) -> tuple[bool, str]:
    """
    驗證工具選擇與計算結果。

    回傳:
      (是否成功, 結果說明)
    """

    actual_required = result.get(
        "calculation_required"
    )

    calculation_results = result.get(
        "calculation_results", []
    )

    if actual_required is not expected_required:
        if expected_required:
            return False, "FALSE NEGATIVE: missed calculation"
        return False, "FALSE POSITIVE: unnecessary calculation"

    if not expected_required:
        if calculation_results:
            return False, "Unexpected calculation results"
        return True, "Correctly skipped calculation"

    if len(calculation_results) != 1:
        return False, "Missing or invalid calculation results"

    calculation = calculation_results[0]

    if calculation.get("operation") != "difference":
        return False, "Incorrect operation"

    if calculation.get("unit") != "million":
        return False, "Incorrect result unit"

    try:
        actual_difference = Decimal(
            str(calculation["result"])
        )

        actual_operands = {
            Decimal(str(
                calculation["operand_a"]["normalized_value"]
            )),
            Decimal(str(
                calculation["operand_b"]["normalized_value"]
            )),
        }

    except (KeyError, ValueError, TypeError):
        return False, "Invalid calculation result structure"

    if actual_operands != {
        Decimal("1046090"),
        Decimal("150188"),
    }:
        return False, "Incorrect calculation operands"

    if actual_difference != EXPECTED_DIFFERENCE:
        return False, "Incorrect difference value"

    return True, "Correct difference calculation"


@pytest.mark.parametrize(
    "case_id, question, expected_required",
    TEST_CASES,
    ids=["TC02", "TC04"],
)
def test_calculator_decision_stability(
    case_id: str,
    question: str,
    expected_required: bool,
):
    """
    使用相同 Evidence，重複測試 Calculator。

    每次都呼叫真正的 LM Studio LLM，
    不使用 Mock。

    單次錯誤不會提前中止，
    方便統計完整測試結果。
    """

    passed = 0
    false_positive = 0
    false_negative = 0
    other_failures = 0
    failures = []

    print(f"\n========== {case_id} ==========")
    print("Question:", question)
    print("Expected calculation:", expected_required)
    print("Repeat count:", REPEAT_COUNT)

    for index in range(1, REPEAT_COUNT + 1):

        state = {
            "standalone_question": question,
            "evidence": FIXED_EVIDENCE,
        }

        try:
            result = calculator(state)

            success, reason = verify_calculator_output(
                result,
                expected_required,
            )

        except Exception as exc:
            success = False
            reason = (
                f"EXCEPTION: {type(exc).__name__}: {exc}"
            )

        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
            failures.append((index, reason))

            if reason.startswith("FALSE POSITIVE"):
                false_positive += 1
            elif reason.startswith("FALSE NEGATIVE"):
                false_negative += 1
            else:
                other_failures += 1

        print(
            f"[{case_id}] Run {index:02d}/"
            f"{REPEAT_COUNT}: {status} - {reason}"
        )

    accuracy = passed / REPEAT_COUNT * 100

    print(f"\n========== {case_id} SUMMARY ==========")
    print(f"Passed: {passed}/{REPEAT_COUNT}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"Other Failures: {other_failures}")
    print("=====================================")

    assert passed == REPEAT_COUNT, (
        f"{case_id}: {passed}/{REPEAT_COUNT} passed; "
        f"failures={failures}"
    )
