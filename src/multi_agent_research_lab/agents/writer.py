"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.
        """

        source_lines = []
        for index, source in enumerate(state.sources, start=1):
            source_lines.append(f"[{index}] {source.title} ({source.url or 'no-url'})")

        with trace_span("agent.writer.run", {"query": state.request.query}) as span:
            response = self._llm_client.complete(
                system_prompt=(
                    "You are a writer agent. Create a clear final answer for technical learners, "
                    "grounded in analysis notes. Include source references like [1], [2]."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n\n"
                    f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
                    f"Research notes:\n{state.research_notes or ''}\n\n"
                    f"Sources:\n{chr(10).join(source_lines)}"
                ),
            )

        state.final_answer = response.content
        logger.info(
            "Writer produced final answer with output_tokens=%s",
            response.output_tokens,
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "agent.writer",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "duration_seconds": span["duration_seconds"],
            },
        )
        return state
