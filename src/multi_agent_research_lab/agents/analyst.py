"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.
        """

        research_notes = state.research_notes or ""
        if not research_notes and state.sources:
            research_notes = "\n".join(source.snippet for source in state.sources)

        with trace_span("agent.analyst.run", {"query": state.request.query}) as span:
            response = self._llm_client.complete(
                system_prompt=(
                    "You are an analyst agent. Extract key claims, compare viewpoints, "
                    "and flag weak or unsupported evidence in concise bullet points."
                ),
                user_prompt=f"Query: {state.request.query}\n\nResearch notes:\n{research_notes}",
            )

        state.analysis_notes = response.content
        logger.info(
            "Analyst produced notes with output_tokens=%s",
            response.output_tokens,
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "agent.analyst",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state
