
from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import agent_graph
from agent_node import check_financial_unit_consistency


PLAN = [
    {
        "task_id": "task_1",
        "company": "台積電",
        "period": "2025Q4",
        "topic": "營收",
        "query": "台積電 2025Q4 營收",
    },
    {
        "task_id": "task_2",
        "company": "聯發科",
        "period": "2025Q4",
        "topic": "營收",
        "query": "聯發科 2025Q4 營收",
    },
]


def make_evidence(task, wrong=False):

    if task["task_id"] == "task_1":
        answer = (
            "100 億"
            if wrong
            else "1,046.09 billion"
        )
        contexts = [
            "NT$ billions. Net Revenue: 1,046.09"
        ]
    else:
        answer = "150 million"
        contexts = ["150 million"]

    return {
        **task,
        "answer": answer,
        "retrieved_contexts": contexts,
        "metadata": [],
        "sources": [],
    }


def initial_state(previous_evidence=None):

    return {
        "question": "比較台積電和聯發科 2025Q4 營收",
        "standalone_question": "",
        "chat_history": [],
        "intent": "",
        "companies": [],
        "periods": [],
        "router_confidence": 0.0,
        "research_plan": [],
        "previous_validated_evidence": (
            previous_evidence or []
        ),
        "reuse_evidence": False,
        "current_task": 0,
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


async def run_scenario(scenario):

    counters = Counter()

    class FakeSubgraph:
        async def ainvoke(self, state):

            task = state["task"]
            task_id = task["task_id"]

            counters[f"worker_{task_id}"] += 1

            failed = (
                scenario == "partial_failure"
                and task_id == "task_2"
            )

            if failed:
                return {
                    **state,
                    "sufficient": False,
                    "retrieval_count": 2,
                    "failure_type": "missing_evidence",
                    "evidence": [],
                }

            evidence = make_evidence(
                task,
                wrong=(
                    scenario == "regeneration"
                    and task_id == "task_1"
                ),
            )

            return {
                **state,
                "sufficient": True,
                "retrieval_count": 0,
                "failure_type": "none",
                "evidence": [evidence],
            }

    def fake_contextualize(state):
        counters["contextualize"] += 1
        return {
            "standalone_question": state["question"]
        }

    def fake_router(state):
        counters["router"] += 1
        return {
            "intent": "peer_comparison",
            "companies": ["台積電", "聯發科"],
            "periods": ["2025Q4"],
            "router_confidence": 1.0,
        }

    def fake_planner(state):
        counters["planner"] += 1
        return {
            "research_plan": [dict(t) for t in PLAN]
        }

    def fake_reuse_checker(state):
        counters["reuse_checker"] += 1

        previous = state.get(
            "previous_validated_evidence", []
        )

        if not previous:
            return {"reuse_evidence": False}

        # 測試 reuse HIT 的路由，不測實際語意匹配
        return {
            "reuse_evidence": True,
            "evidence": previous,
        }

    def fake_global_checker(state):
        counters["global_checker"] += 1

        issues = []
        for item in state["evidence"]:
            issues.extend(
                check_financial_unit_consistency(
                    answer=item["answer"],
                    retrieved_contexts=item["retrieved_contexts"],
                )
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

    def fake_calculator(state):
        counters["calculator"] += 1

        return {
            "calculation_required": False,
            "calculation_results": [],
        }

    def fake_report_writer(state):
        counters["report_writer"] += 1

        assert state["sufficient"] is True

        return {
            "final_answer": "Validated financial report"
        }

    # Step 5C 的正式 Answer Regenerator 使用此 Mock
    fake_completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=SimpleNamespace(
                        answer="1,046.09 billion"
                    )
                )
            )
        ]
    )

    previous = None
    if scenario == "reuse":
        previous = [
            make_evidence(task)
            for task in PLAN
        ]

    # 重要：先 Patch，再建立正式 Graph。
    # LangGraph 在 add_node 時會保存函式參照。
    with (
        patch.multiple(
            agent_graph,
            contextualize_question=fake_contextualize,
            intent_router=fake_router,
            research_planner=fake_planner,
            evidence_reuse_checker=fake_reuse_checker,
            evidence_checker=fake_global_checker,
            calculator=fake_calculator,
            report_writer=fake_report_writer,
        ),
        patch(
            "fanout_nodes.build_research_task_graph",
            return_value=FakeSubgraph(),
        ),
        patch(
            "agent_node.client.chat.completions.parse",
            return_value=fake_completion,
        ) as mock_llm,
    ):

        # 使用專案正式的 build_agent_graph()
        app = agent_graph.build_agent_graph()

        result = await app.ainvoke(
            initial_state(previous),
            config={"recursion_limit": 30},
        )

        llm_calls = mock_llm.call_count

    print(f"\n===== {scenario} =====")
    print("Counters:", dict(counters))
    print("Evidence:", len(result["evidence"]))
    print("Sufficient:", result["sufficient"])
    print("Regeneration:", result["regeneration_count"])
    print("Final answer:", result["final_answer"])

    if scenario == "normal":

        assert counters["worker_task_1"] == 1
        assert counters["worker_task_2"] == 1
        assert counters["global_checker"] == 1
        assert counters["calculator"] == 1
        assert counters["report_writer"] == 1

        assert result["sufficient"] is True
        assert len(result["evidence"]) == 2
        assert len(result["task_results"]) == 2
        assert result["final_answer"] == (
            "Validated financial report"
        )
        assert llm_calls == 0

    elif scenario == "partial_failure":

        assert counters["worker_task_1"] == 1
        assert counters["worker_task_2"] == 1

        # Fan-in 應直接走 Failure Writer。
        assert counters["global_checker"] == 0
        assert counters["calculator"] == 0
        assert counters["report_writer"] == 0

        assert result["sufficient"] is False
        assert len(result["evidence"]) == 1
        assert len(result["task_results"]) == 2

        assert "台積電" in result["final_answer"]
        assert "聯發科" in result["final_answer"]
        assert "已重試 2 次" in result["final_answer"]
        assert llm_calls == 0

    elif scenario == "regeneration":

        assert counters["worker_task_1"] == 1
        assert counters["worker_task_2"] == 1
        assert counters["global_checker"] == 2
        assert counters["calculator"] == 1
        assert counters["report_writer"] == 1

        assert result["regeneration_count"] == 1
        assert result["sufficient"] is True
        assert len(result["evidence"]) == 2

        evidence_map = {
            item["task_id"]: item
            for item in result["evidence"]
        }

        assert (
            evidence_map["task_1"]["answer"]
            == "1,046.09 billion"
        )
        assert (
            evidence_map["task_2"]["answer"]
            == "150 million"
        )

        assert llm_calls == 1

    elif scenario == "reuse":

        assert counters["worker_task_1"] == 0
        assert counters["worker_task_2"] == 0
        assert counters["global_checker"] == 1
        assert result["sufficient"] is True

        assert counters["calculator"] == 1
        assert counters["report_writer"] == 1

        assert result["reuse_evidence"] is True
        assert len(result["evidence"]) == 2
        assert result["final_answer"] == (
            "Validated financial report"
        )
        assert llm_calls == 0

    print(f"[PASS] {scenario}")


async def main():

    for scenario in (
        "normal",
        "partial_failure",
        "regeneration",
        "reuse",
    ):
        await run_scenario(scenario)

    print("\n[PASS] All Step 6 integration tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
