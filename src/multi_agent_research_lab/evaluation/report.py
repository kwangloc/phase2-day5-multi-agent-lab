"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Errors | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citations = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        lines.append(
            "| "
            f"{item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{citations} | {item.error_count} | {item.notes} |"
        )

    if metrics:
        fastest = min(metrics, key=lambda item: item.latency_seconds)
        best_quality = max(metrics, key=lambda item: item.quality_score or 0)
        cost_known = [item for item in metrics if item.estimated_cost_usd is not None]

        lines.extend([
            "",
            "## Summary",
            "",
            f"- Fastest run: {fastest.run_name} ({fastest.latency_seconds:.2f}s)",
            (
                "- Highest quality: "
                f"{best_quality.run_name} ({(best_quality.quality_score or 0):.1f}/10)"
            ),
        ])
        if cost_known:
            cheapest = min(cost_known, key=lambda item: item.estimated_cost_usd or 0.0)
            lines.append(
                "- Lowest estimated cost: "
                f"{cheapest.run_name} (${(cheapest.estimated_cost_usd or 0.0):.4f})"
            )

    return "\n".join(lines) + "\n"
