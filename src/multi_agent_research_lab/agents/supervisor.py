"""Supervisor / router skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @staticmethod
    def _has_critic_result(state: ResearchState) -> bool:
        return any(result.agent == "critic" for result in state.agent_results)

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.
        """

        with trace_span(
            "supervisor.run",
            {
                "iteration": state.iteration,
                "errors": len(state.errors),
            },
        ) as span:
            if state.iteration >= self._settings.max_iterations or len(state.errors) >= 3:
                route = "done"
            elif not state.sources or not state.research_notes:
                route = "researcher"
            elif not state.analysis_notes:
                route = "analyst"
            elif not state.final_answer:
                route = "writer"
            elif not self._has_critic_result(state):
                route = "critic"
            else:
                route = "done"

        state.record_route(route)
        logger.info(
            "Supervisor routed to %s at iteration=%s errors=%s",
            route,
            state.iteration,
            len(state.errors),
        )
        state.add_trace_event(
            "supervisor.route",
            {
                "route": route,
                "iteration": state.iteration,
                "errors": len(state.errors),
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state
