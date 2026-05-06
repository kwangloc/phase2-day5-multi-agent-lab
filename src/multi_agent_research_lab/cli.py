"""Command-line entrypoint for the lab starter."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline_query(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent research assistant. Provide a concise technical summary "
            "for the given query and include key considerations."
        ),
        user_prompt=query,
    )
    state.final_answer = response.content
    state.add_trace_event(
        "baseline.complete",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    return state


def _run_multi_agent_query(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def _artifact_store() -> LocalArtifactStore:
    repo_root = Path(__file__).resolve().parents[2]
    return LocalArtifactStore(root=repo_root / "reports")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _export_state_trace(run_name: str, state: ResearchState) -> Path:
    relative_path = f"traces/{run_name}_trace_{_timestamp()}.json"
    payload = {
        "run_name": run_name,
        "request": state.request.model_dump(mode="json"),
        "route_history": state.route_history,
        "trace": state.trace,
        "errors": state.errors,
        "state": state.model_dump(mode="json"),
    }
    return _artifact_store().write_json(relative_path, payload)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    state = _run_baseline_query(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    try:
        result = _run_multi_agent_query(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    trace_path = _export_state_trace("multi_agent", result)
    console.print(result.model_dump_json(indent=2))
    console.print(f"Saved trace to {trace_path}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Benchmark baseline vs multi-agent and write a markdown report."""

    _init()
    benchmark_runs = [
        run_benchmark("baseline", query, _run_baseline_query),
        run_benchmark("multi-agent", query, _run_multi_agent_query),
    ]
    states = {
        metrics.run_name: state
        for state, metrics in benchmark_runs
    }
    metrics = [item[1] for item in benchmark_runs]
    report = render_markdown_report(metrics)
    path = _artifact_store().write_text("benchmark_report.md", report)
    trace_path = _artifact_store().write_json(
        f"traces/benchmark_{_timestamp()}.json",
        {
            "query": query,
            "metrics": [metric.model_dump(mode="json") for metric in metrics],
            "states": {name: state.model_dump(mode="json") for name, state in states.items()},
        },
    )

    console.print(Panel.fit(report, title="Benchmark Report"))
    console.print(f"Saved report to {path}")
    console.print(f"Saved trace bundle to {trace_path}")


if __name__ == "__main__":
    app()
