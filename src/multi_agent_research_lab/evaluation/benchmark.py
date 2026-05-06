"""Benchmark helpers for single-agent vs multi-agent evaluation."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _estimate_cost_from_state(state: ResearchState) -> float | None:
    total_cost = 0.0
    has_cost = False

    for result in state.agent_results:
        raw_cost = result.metadata.get("cost_usd")
        if isinstance(raw_cost, int | float):
            total_cost += float(raw_cost)
            has_cost = True

    for event in state.trace:
        raw_cost = event.get("payload", {}).get("cost_usd")
        if isinstance(raw_cost, int | float):
            total_cost += float(raw_cost)
            has_cost = True

    return total_cost if has_cost else None


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.sources:
        return None

    answer = state.final_answer or ""
    citation_ids = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    if not citation_ids:
        return 0.0

    cited_sources = sum(1 for index in range(1, len(state.sources) + 1) if index in citation_ids)
    return min(1.0, cited_sources / len(state.sources))


def _quality_score(state: ResearchState, citation_coverage: float | None) -> float:
    score = 0.0

    if state.final_answer:
        score += 4.0
        answer_length = len(state.final_answer.split())
        if answer_length >= 80:
            score += 1.0
    if state.analysis_notes:
        score += 2.0
    if state.research_notes:
        score += 1.0
    if state.sources:
        score += min(2.0, len(state.sources) * 0.4)
    if citation_coverage is not None:
        score += citation_coverage
    if state.errors:
        score -= min(3.0, float(len(state.errors)))

    return max(0.0, min(10.0, round(score, 2)))


def _notes_summary(state: ResearchState, citation_coverage: float | None) -> str:
    parts = [f"sources={len(state.sources)}", f"errors={len(state.errors)}"]
    if citation_coverage is not None:
        parts.append(f"citations={citation_coverage:.0%}")
    if state.route_history:
        parts.append("route=" + "->".join(state.route_history))
    return ", ".join(parts)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive lightweight quality/cost metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    citation_coverage = _citation_coverage(state)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost_from_state(state),
        quality_score=_quality_score(state, citation_coverage),
        citation_coverage=citation_coverage,
        error_count=len(state.errors),
        notes=_notes_summary(state, citation_coverage),
    )
    return state, metrics
