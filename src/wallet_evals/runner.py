"""Run cases x adapter x repeats and score each attempt."""
from __future__ import annotations

from dataclasses import dataclass

from wallet_evals.adapters.base import ModelAdapter
from wallet_evals.schema import Case
from wallet_evals.scorer import score_case


@dataclass
class CaseResult:
    case_id: str
    repeat: int
    score: int
    model: str
    level: str
    protocol: str
    query_type: str | None
    difficulty: str
    language: str
    requires: list[str]


def run_eval(cases: list[Case], adapter: ModelAdapter, *, repeats: int = 1) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        for repeat in range(repeats):
            turn = adapter.run(case.user_message)
            results.append(CaseResult(
                case_id=case.id,
                repeat=repeat,
                score=score_case(case, turn),
                model=adapter.model,
                level=case.level,
                protocol=case.protocol,
                query_type=case.query_type,
                difficulty=case.difficulty,
                language=case.language,
                requires=list(case.requires),
            ))
    return results
