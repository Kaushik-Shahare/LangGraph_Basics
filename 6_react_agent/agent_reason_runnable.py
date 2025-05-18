from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

import os
key = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=api_key)
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key = key)

search_tool = TavilySearchResults(search_depth='basic')


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time formatted according to the provided format string."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
