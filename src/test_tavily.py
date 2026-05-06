from tavily import TavilyClient  # type: ignore[import-untyped]

tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("Who is Leo Messi?")

print(response)