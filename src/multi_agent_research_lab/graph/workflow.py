"""LangGraph workflow implementation."""

import logging
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class WorkflowPayload(TypedDict):
    state: ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._supervisor = SupervisorAgent(settings=self._settings)
        self._agents = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent(),
        }

    def _run_supervisor(self, payload: WorkflowPayload) -> WorkflowPayload:
        return {"state": self._supervisor.run(payload["state"])}

    def _run_named_agent(self, name: str, payload: WorkflowPayload) -> WorkflowPayload:
        state = payload["state"]
        agent = self._agents.get(name)
        if agent is None:
            logger.error("Unknown workflow route encountered: %s", name)
            state.errors.append(f"Unknown route: {name}")
            state.add_trace_event("workflow.error", {"error": f"Unknown route: {name}"})
            return {"state": state}

        try:
            updated = agent.run(state)
        except Exception as exc:
            logger.exception("Agent execution failed for route=%s", name)
            state.errors.append(f"{name} failed: {exc}")
            state.add_trace_event("workflow.error", {"route": name, "error": str(exc)})
            updated = state

        return {"state": updated}

    @staticmethod
    def _route_from_supervisor(payload: WorkflowPayload) -> str:
        state = payload["state"]
        if not state.route_history:
            return "done"
        return state.route_history[-1]

    def build(self) -> Any:
        """Create and compile the LangGraph workflow."""

        graph = StateGraph(WorkflowPayload)
        graph.add_node("supervisor", cast(Any, self._run_supervisor))
        graph.add_node(
            "researcher",
            cast(Any, lambda payload: self._run_named_agent("researcher", payload)),
        )
        graph.add_node(
            "analyst",
            cast(Any, lambda payload: self._run_named_agent("analyst", payload)),
        )
        graph.add_node(
            "writer",
            cast(Any, lambda payload: self._run_named_agent("writer", payload)),
        )
        graph.add_node(
            "critic",
            cast(Any, lambda payload: self._run_named_agent("critic", payload)),
        )

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return final state."""

        with trace_span("workflow.run", {"query": state.request.query}) as span:
            app = cast(Any, self.build())
            result = cast(WorkflowPayload, app.invoke({"state": state}))
        final_state = result["state"]
        logger.info(
            "Workflow completed with iterations=%s errors=%s",
            final_state.iteration,
            len(final_state.errors),
        )
        final_state.add_trace_event(
            "workflow.run",
            {
                "duration_seconds": span["duration_seconds"],
                "iterations": final_state.iteration,
                "errors": len(final_state.errors),
            },
        )

        if final_state.iteration >= self._settings.max_iterations:
            final_state.add_trace_event("workflow.stop", {"reason": "max_iterations"})

        return final_state
