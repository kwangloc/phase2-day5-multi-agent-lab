"""Optional critic agent skeleton for bonus work."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.
        """

        final_answer = state.final_answer or ""
        research_notes = state.research_notes or ""

        with trace_span("agent.critic.run", {"query": state.request.query}) as span:
            response = self._llm_client.complete(
                system_prompt=(
                    "You are a critic agent. Review the final answer for citation coverage, "
                    "unsupported claims, and missing caveats. Return concise bullets."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n\n"
                    f"Final answer:\n{final_answer}\n\n"
                    f"Research notes:\n{research_notes}"
                ),
            )

        logger.info(
            "Critic reviewed final answer with output_tokens=%s",
            response.output_tokens,
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "duration_seconds": span["duration_seconds"],
                },
            )
        )
        state.add_trace_event(
            "agent.critic",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state
