from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You re expert AI researcher.
            Current Time: {time}
            1. {first_instruction}
            2. Reflext and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries seperately** for researching improvements. Do not include them inside the reflection.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]

).partial(
    # Pre populate the variable
    time=lambda: datetime.datetime.now().isoformat(),
)


llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

first_responder_prompt_templete = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

# bind_tools help us provide tools to the llm
# tools: Array of all the tools provided to LLM
# tool_choice: force the llm to use a particular tool
first_responder_chain = first_responder_prompt_templete | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")


validator = PydanticToolsParser(tools=[AnswerQuestion])


revise_instruction = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a 'References' section to the bottom of your answer (which does not count towards the word limit). In from of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words."""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})

print(response)
