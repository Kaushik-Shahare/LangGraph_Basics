from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ( 
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for te user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        # previous convo history
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influncer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including request for length, virality, style, etc.",
        ),
        # previous convo history
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Pipe the prompt with llm, is part of LangChain’s LCEL (LangChain Expression Language). It’s a powerful and clean way to build chains by piping the output of one component into the next — similar to how Unix pipes work.
# Takes the output of reflection_prompt and Pass it as input to the llm.
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
