from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

import os
key = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=api_key)
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key = key)

#llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
search_tool = TavilySearchResults(search_depth='basic')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time formatted according to the provided format string."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# result = llm.invoke("Give me a fact about cats")

boiler_pates_notes = """
Generate structured, concise, and precise study notes in Markdown format for the topic “{TOPIC}” with the following format:
- Avoid adding starting and ending "```" for the markdown. Just return the Text inside it as markdown.

⸻

{TOPIC}

Types of {TOPIC}

(If applicable — list types with short descriptions.)

Detailed Notes

Explain the concept clearly and concisely using bullet points or small paragraphs. Cover essential subtopics, use simple language, and make it beginner-friendly but informative.

Include examples where appropriate (preferably in code blocks or clear text examples).

Key Points
	•	Summarize important facts, formulas, characteristics, or core ideas.

Advance Tips and Best Practices                                             e
	•	Share optimization tips, best practices, and common pitfalls to avoid.

Also include the following:
	•	Relevant image links using trusted sources such as GeeksforGeeks, W3Schools, and LeetCode (in that order of priority). Embed them in markdown format:
![alt text](image_url)
	•	Use clean and consistent formatting — no emojis, no unnecessary elaboration, no fluff.
	•	Reference examples or explanations from the websites above wherever applicable.


Topic : 
"""


#agent.invoke("Give me a twet about today's wether in Vadodara.")            e
# agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant.")
agent.invoke(boiler_pates_notes + "Give me a detailed Markdown notes about tree in python in MarkDown Format. Use GeeksForGeeks for reference and w3schools for python reference. Also Give Advance Tips for best practices for tree. Also make sure to Give concise and precise notes with working examples and no Emoji in MarkDown.")

