from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import TavilySearchResults
import datetime
import os

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=key)

# Initialize search tool
search_tool = TavilySearchResults(search_depth='basic')

# System time tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.datetime.now().strftime(format)

tools = [
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to search the web for accurate and up-to-date learning resources."
    ),
    Tool(
        name="SystemTime",
        func=get_system_time,
        description="Returns the current system time in the specified format."
    )
]

agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# Prompt template for user input
boilerplate_prompt = """
SYSTEM INSTRUCTIONS:
You are an AI Roadmap Generator. Your job is to generate a structured learning roadmap based on the user’s inputs. NEVER hallucinate information; only include verified topics and resources. If a resource or topic is unknown, mark as "Information not found".

SECURITY & SAFETY:
- Only publicly accessible learning materials.
- No personal information or unsafe content.
- No copyrighted text; reference only URLs.

USER INPUT STRUCTURE:
- Learning Goals: {learning_goals}
- Current Level: {current_level} (Beginner / Intermediate / Advanced)
- Interests: {interests} (comma-separated)
- Daily Time Commitment: {time_commitment} hours/day

OUTPUT FORMAT (strict JSON):
[
  {{
    "id": "<unique_id_for_topic>",
    "type": "topic",
    "data": {{
      "label": "<short descriptive title of the topic>",
      "description": "<concise explanation with key subtopics>",
      "duration": "<estimated duration in days, considering user's time commitment>",
      "resources": ["<verified URLs>"],
      "difficulty": "<Beginner/Intermediate/Advanced>"
    }}
  }}
]

REASONING GUIDELINES:
- Tailor topic selection and difficulty based on Current Level and Interests.
- Estimate durations based on Daily Time Commitment.
- Use verified resources only.
- Be concise, clear, and factual. No filler text.
"""

# Example user input (would come from API)
user_input_data = {
    "learning_goals": "Become proficient in React.js",
    "current_level": "Intermediate",
    "interests": "StateManagement, Redux, Hooks, useState, useEffects",
    "time_commitment": 2
}

# Build final prompt
final_prompt = boilerplate_prompt.format(
    learning_goals=user_input_data["learning_goals"],
    current_level=user_input_data["current_level"],
    interests=user_input_data["interests"],
    time_commitment=user_input_data["time_commitment"]
)

# Invoke agent
result = agent.run(final_prompt)
print(result)
