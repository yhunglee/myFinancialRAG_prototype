
from agent_node import failure_report_writer


def test_partial_task_failure():
    """
    Case A:
    台積電成功，聯發科失敗。
    """

    state = {
        "failure_type": "missing_evidence",
        "task_results": [
            {
                "task_id": "task_1",
                "company": "台積電",
                "period": "2025Q4",
                "topic": "營收",
                "sufficient": True,
                "retrieval_count": 0,
                "failure_type": "none",
            },
            {
                "task_id": "task_2",
                "company": "聯發科",
                "period": "2025Q4",
                "topic": "營收",
                "sufficient": False,
                "retrieval_count": 2,
                "failure_type": "missing_evidence",
            },
        ],
    }

    result = failure_report_writer(state)
    answer = result["final_answer"]

    print("\n===== Case A: Partial Task Failure =====")
    print(answer)

    assert "台積電" in answer
    assert "聯發科" in answer
    assert "證據檢查通過" in answer
    assert "缺少必要的財報證據" in answer
    assert "已重試 2 次" in answer
    assert "不提供完整的財務比較結論" in answer

    print("[PASS] Partial task failure reported.")


def test_weak_evidence_failure():
    """
    Case B:
    個別任務檢索到資料，
    但品質仍不足。
    """

    state = {
        "failure_type": "missing_evidence",
        "task_results": [
            {
                "task_id": "task_1",
                "company": "台積電",
                "period": "2025Q4",
                "topic": "毛利率",
                "sufficient": False,
                "retrieval_count": 2,
                "failure_type": "weak_evidence",
            },
        ],
    }

    result = failure_report_writer(state)
    answer = result["final_answer"]

    print("\n===== Case B: Weak Evidence =====")
    print(answer)

    assert "台積電" in answer
    assert "毛利率" in answer
    assert "證據品質不足" in answer

    print("[PASS] Weak evidence reported.")


def test_global_validation_failure():
    """
    Case C:
    所有 Task 已成功，
    但全域答案一致性檢查失敗。
    """

    state = {
        "failure_type": "answer_not_supported",
        "regeneration_count": 2,
        "task_results": [
            {
                "task_id": "task_1",
                "sufficient": True,
            },
            {
                "task_id": "task_2",
                "sufficient": True,
            },
        ],
    }

    result = failure_report_writer(state)
    answer = result["final_answer"]

    print("\n===== Case C: Global Failure =====")
    print(answer)

    assert "無法通過" in answer
    assert "不提供未經驗證的財務結論" in answer

    print("[PASS] Global failure reported.")


def test_missing_information():
    """
    Case D:
    全域檢查指出具體缺漏。
    """

    state = {
        "failure_type": "missing_evidence",
        "task_results": [],
        "missing_information": [
            "聯發科 2025Q4 營收",
        ],
    }

    result = failure_report_writer(state)
    answer = result["final_answer"]

    print("\n===== Case D: Missing Information =====")
    print(answer)

    assert "資料不足" in answer
    assert "聯發科 2025Q4 營收" in answer

    print("[PASS] Missing information reported.")


def main():
    test_partial_task_failure()
    test_weak_evidence_failure()
    test_global_validation_failure()
    test_missing_information()

    print("\n[PASS] All Step 5E report tests passed.")


if __name__ == "__main__":
    main()
