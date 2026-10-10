
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from langgraph.graph import StateGraph, START, END

from agent_state import FinancialResearchState
from agent_graph import (
    route_after_global_evidence_check,
    MAX_REGENERATION_ATTEMPTS,
)
from agent_node import (
    answer_regenerator,
    failure_report_writer,
    check_financial_unit_consistency,
)


def build_test_graph(checker, counters):
    """
    建立測試專用 Graph。
    使用正式的 Regenerator、Failure Writer 和 Routing。
    """

    def tracked_regenerator(state):
        counters["regenerator"] += 1
        print(
            "[Regenerator]",
            "call =", counters["regenerator"],
        )
        return answer_regenerator(state)

    def tracked_failure_writer(state):
        counters["failure_writer"] += 1
        return failure_report_writer(state)

    def fake_calculator(state):
        counters["calculator"] += 1
        return {"calculation_results": []}

    graph = StateGraph(FinancialResearchState)

    graph.add_node("evidence_checker", checker)
    graph.add_node("answer_regenerator", tracked_regenerator)
    graph.add_node("calculator", fake_calculator)
    graph.add_node("failure_report_writer", tracked_failure_writer)

    graph.add_edge(START, "evidence_checker")

    graph.add_conditional_edges(
        "evidence_checker",
        route_after_global_evidence_check,
        {
            "proceed": "calculator",
            "regenerate_answer": "answer_regenerator",
            "stop": "failure_report_writer",
        },
    )

    graph.add_edge(
        "answer_regenerator",
        "evidence_checker",
    )

    graph.add_edge("calculator", END)
    graph.add_edge("failure_report_writer", END)

    return graph.compile()


def create_initial_state():
    return {
        "evidence": [
            {
                "task_id": "task_1",
                "query": "台積電 2025Q4 營收",
                "answer": "100 億",
                "retrieved_contexts": [
                    "NT$ billions. Net Revenue: 1,046.09"
                ],
            },
            {
                "task_id": "task_2",
                "query": "聯發科 2025Q4 營收",
                "answer": "150 million",
                "retrieved_contexts": [
                    "150 million"
                ],
            },
        ],
        "regeneration_count": 0,
        "sufficient": False,
        "failure_type": "none",
        "next_action": "proceed",
        "unsupported_answer": [],
    }


def test_regeneration_limit():
    """
    Case A:
    LLM 修正兩次，但每次仍給出錯誤答案。
    Graph 必須達到上限後安全停止。
    """

    counters = {
        "checker": 0,
        "regenerator": 0,
        "failure_writer": 0,
        "calculator": 0,
    }

    def fake_evidence_checker(state):
        counters["checker"] += 1

        issues = []

        for item in state["evidence"]:
            issues.extend(
                check_financial_unit_consistency(
                    answer=item["answer"],
                    retrieved_contexts=item["retrieved_contexts"],
                )
            )

        print(
            "[Checker]",
            "call =", counters["checker"],
            "regeneration_count =", state.get("regeneration_count", 0),
        )

        if issues:
            return {
                "sufficient": False,
                "failure_type": "answer_not_supported",
                "next_action": "regenerate_answer",
                "unsupported_answer": issues,
            }

        return {
            "sufficient": True,
            "failure_type": "none",
            "next_action": "proceed",
            "unsupported_answer": [],
        }

    app = build_test_graph(
        fake_evidence_checker,
        counters,
    )

    # 故意讓 LLM 每次都回傳不正確的答案。
    fake_completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=SimpleNamespace(
                        answer="200 億"
                    )
                )
            )
        ]
    )

    with patch(
        "agent_node.client.chat.completions.parse",
        return_value=fake_completion,
    ) as mock_llm:

        result = app.invoke(create_initial_state())

        assert mock_llm.call_count == MAX_REGENERATION_ATTEMPTS

    print("\n===== Case A: Regeneration Limit =====")
    print("Counters:", counters)
    print("Regeneration count:", result["regeneration_count"])
    print("Final answer:", result["final_answer"])

    assert counters["checker"] == 3
    assert counters["regenerator"] == 2
    assert counters["failure_writer"] == 1
    assert counters["calculator"] == 0

    assert result["regeneration_count"] == 2
    assert result["sufficient"] is False
    assert result["failure_type"] == "answer_not_supported"
    assert result["next_action"] == "regenerate_answer"

    # 即使修正失敗，也不應累積重複 Evidence。
    assert len(result["evidence"]) == 2

    evidence_map = {
        item["task_id"]: item
        for item in result["evidence"]
    }

    assert evidence_map["task_1"]["answer"] == "200 億"
    assert evidence_map["task_2"]["answer"] == "150 million"

    # 確認有產生失敗報告，而非未經驗證的正常結論。
    assert result["final_answer"]
    assert "無法通過證據一致性驗證" in result["final_answer"]

    print("[PASS] Regeneration limit enforced.")


def test_global_retrieval_disabled():
    """
    Case B:
    全域 Checker 要求 retrieve_again。
    由於舊版 Global Retrieval 已停用，
    Graph 必須直接進入失敗處理。
    """

    counters = {
        "checker": 0,
        "regenerator": 0,
        "failure_writer": 0,
        "calculator": 0,
    }

    def fake_evidence_checker(state):
        counters["checker"] += 1

        print(
            "[Checker]",
            "call =", counters["checker"],
            "regeneration_count =", state.get("regeneration_count", 0),
        )

        return {
            "sufficient": False,
            "failure_type": "missing_evidence",
            "next_action": "retrieve_again",
            "missing_information": [
                "聯發科 2025Q4 營收"
            ],
        }

    app = build_test_graph(
        fake_evidence_checker,
        counters,
    )

    # 若錯誤呼叫 LLM，測試會直接失敗。
    with patch(
        "agent_node.client.chat.completions.parse",
        side_effect=AssertionError(
            "LLM should not be called"
        ),
    ) as mock_llm:

        result = app.invoke(create_initial_state())

        mock_llm.assert_not_called()

    print("\n===== Case B: Global Retrieval Disabled =====")
    print("Counters:", counters)
    print("Final answer:", result["final_answer"])

    assert counters["checker"] == 1
    assert counters["regenerator"] == 0
    assert counters["failure_writer"] == 1
    assert counters["calculator"] == 0

    assert result["sufficient"] is False
    assert result["failure_type"] == "missing_evidence"
    assert result["next_action"] == "retrieve_again"
    assert result["regeneration_count"] == 0

    assert result["final_answer"]

    print("[PASS] Global retrieval safely disabled.")


def main():
    test_regeneration_limit()
    test_global_retrieval_disabled()

    print("\n[PASS] All Step 5D tests passed.")


if __name__ == "__main__":
    main()
