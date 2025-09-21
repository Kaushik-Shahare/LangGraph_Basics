import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Define the Tools
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.

If you are asked a question, you should use the `internet_search` tool to find the answer.
"""

# 2. Create the LLM
llm = ChatGoogleGenerativeAI(api_key=os.environ["GEMINI_API_KEY"], model="gemini-1.5-flash")

# Create the agent
agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
    model=llm,
)

# Invoke the agent
result = agent.invoke({"messages": [("user", "Use the internet_search tool to find out what is causing the global warming and what human activites contribute towards it.")]})

print("---FINAL RESULT---")
print(result['messages'][-1].content)
