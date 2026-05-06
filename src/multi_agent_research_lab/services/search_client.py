"""Search client abstraction for ResearcherAgent."""

import json
import logging
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with deterministic fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @staticmethod
    def _mock_results(query: str, max_results: int) -> list[SourceDocument]:
        templates = [
            ("Overview", "High-level concepts, definitions, and context."),
            ("Methods", "Common approaches, system design choices, and trade-offs."),
            ("Evaluation", "Benchmarks, metrics, and observed limitations."),
            ("Production", "Deployment constraints, reliability, and guardrails."),
            ("Future Work", "Open research questions and practical next steps."),
        ]

        compact_query = " ".join(query.split())
        results: list[SourceDocument] = []
        for idx, (title, snippet) in enumerate(templates[: max(1, max_results)], start=1):
            results.append(
                SourceDocument(
                    title=f"{compact_query} - {title}",
                    url=f"https://example.org/{compact_query.replace(' ', '-').lower()}/{idx}",
                    snippet=snippet,
                    metadata={"provider": "mock", "rank": idx},
                )
            )
        return results

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        if not self._settings.tavily_api_key:
            return []

        with trace_span("search.tavily", {"max_results": max_results}) as span:
            request = Request(
                url="https://api.tavily.com/search",
                method="POST",
                data=json.dumps(
                    {
                        "api_key": self._settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urlopen(request, timeout=self._settings.timeout_seconds) as resp:  # nosec: B310
                payload = json.loads(resp.read().decode("utf-8"))

        results: list[SourceDocument] = []
        for idx, item in enumerate(payload.get("results", []), start=1):
            results.append(
                SourceDocument(
                    title=str(item.get("title") or f"Result {idx}"),
                    url=item.get("url"),
                    snippet=str(item.get("content") or ""),
                    metadata={"provider": "tavily", "rank": idx},
                )
            )
        logger.info(
            "Search provider=tavily returned %s results in %.3fs",
            len(results),
            float(span["duration_seconds"] or 0.0),
        )
        return results

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.
        """

        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []

        try:
            real_results = self._tavily_search(normalized_query, max_results=max_results)
            if real_results:
                logger.info("Search using Tavily provider")
                return real_results
        except Exception as exc:
            # Fallback is intentional for offline lab execution.
            logger.warning("Search fallback to mock due to provider error: %s", exc)

        logger.info("Search using mock provider")
        return self._mock_results(normalized_query, max_results=max_results)
