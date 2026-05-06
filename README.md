# Lab 20: Multi-Agent Research System

A production-grade multi-agent research assistant built with **LangGraph**, featuring a Supervisor/Router that orchestrates Researcher, Analyst, Writer, and Critic agents. Includes single-agent baseline, end-to-end benchmarking, and artifact tracing.

## Architecture

```text
User Query
   │
   ▼
Supervisor / Router
   ├──► Researcher Agent  → sources + research_notes  (Tavily or mock search)
   ├──► Analyst Agent     → analysis_notes            (LLM structured insights)
   ├──► Writer Agent      → final_answer              (LLM grounded response)
   └──► Critic Agent      → review bullets            (LLM fact-check)
   │
   ▼
Benchmark Report + Trace JSON
```

## Project Structure

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/          # Supervisor, Researcher, Analyst, Writer, Critic
│   ├── core/            # Config, state, schemas, errors
│   ├── graph/           # LangGraph workflow
│   ├── services/        # LLM client (OpenAI + offline), search client (Tavily + mock)
│   ├── evaluation/      # Benchmark runner + Markdown report renderer
│   ├── observability/   # Logging + JSON tracing
│   └── cli.py           # CLI entrypoint (baseline / multi-agent / benchmark)
├── configs/             # YAML configs
├── docs/                # Lab guide, design template, peer review rubric
├── reports/             # Generated benchmark reports and trace JSON files
├── tests/               # Unit tests
├── .env.example         # Environment variable template
├── pyproject.toml       # Python project config
├── Dockerfile           # Containerized dev/runtime
└── Makefile             # Common dev commands
```

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Configure API keys

Open `.env` and fill in the required keys:

```bash
OPENAI_API_KEY=...       # required for real LLM responses
TAVILY_API_KEY=...       # optional; falls back to mock search
LANGSMITH_API_KEY=...    # optional; for LangSmith tracing
```

### 3. Run tests

```bash
make test
python -m multi_agent_research_lab.cli --help
```

### 4. Run single-agent baseline

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### 5. Run multi-agent workflow

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Saves a trace JSON to `reports/traces/`.

### 6. Benchmark baseline vs multi-agent

```bash
python -m multi_agent_research_lab.cli benchmark \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Writes `reports/benchmark_report.md` and a trace bundle JSON.

## Benchmark Results

Latest results from [`reports/benchmark_report.md`](reports/benchmark_report.md):

| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Errors |
|---|---:|---:|---:|---:|---:|
| baseline | 10.80 | $0.0003 | 5.0/10 | — | 0 |
| multi-agent | 26.61 | $0.0026 | 10.0/10 | 80% | 0 |

**Key tradeoffs:** Multi-agent produces significantly higher quality and citation coverage at ~2.5× latency and ~9× cost. Use single-agent for low-latency, low-cost queries; use multi-agent for research-grade output.

## Design Principles

- Clear separation between `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- No API keys hardcoded; all secrets loaded from `.env` via `pydantic-settings`.
- All inputs/outputs use Pydantic schemas for type safety.
- Agents cannot run indefinitely: `max_iterations` and `timeout_seconds` enforced by Supervisor.
- Retry with exponential backoff on LLM calls via `tenacity`.
- Deterministic offline fallback when no API key is configured (safe for CI).
- Benchmark report generated from actual run metrics, not manual demo output.

## Guardrails

| Guardrail | Implementation |
|---|---|
| Max iterations | `Settings.max_iterations` (default 6), checked by Supervisor |
| Timeout | `Settings.timeout_seconds` (default 60s), passed to OpenAI client |
| LLM retry | `tenacity` — 3 attempts, exponential backoff 1–8s |
| Error accumulation | `state.errors` — Supervisor halts at 3+ errors |
| Offline fallback | `LLMClient` and `SearchClient` return deterministic mock responses |

## Exit Ticket

**Q: When should you use multi-agent?**
When the task has distinct, separable subtasks (search → analyze → write) and output quality justifies extra latency/cost. Multi-agent scored 10/10 vs 5/10 for baseline on a research query.

**Q: When should you NOT use multi-agent?**
When latency or cost is a hard constraint, or when the task is simple enough that a single LLM call suffices. Baseline was 2.5× faster and 9× cheaper.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
