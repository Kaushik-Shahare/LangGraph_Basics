from typing import List, Annotated, Sequence, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

tavily_search = TavilySearchResults(max_results=3)

python_repl_tool = PythonREPLTool()

print(python_repl_tool.invoke("x = 5; print(x)"))

class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
        "'enhancer' for content enhancement, 'researcher' for information gathering, or 'coder' for code generation."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task towards completion."
    )

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    system_prompt = ("""
    You are a workflow supervisor managing a team of heree specialists: an Enhancer, a Researcher, and a Coder. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current task requirements and the progress made so far. Provide a clear, concise rationale for your choice, ensuring that the selected specialist is best suited to advance the task towards completion.

    **Team Members:**
    1. **Prompt Enhancer**: Focuses on refining and improving the quality of the prompt or task description.
    2. **Researcher**: Gathers relevant information, data, or insights to support the task at hand.
    3. **Coder**: Develops code or technical solutions based on the requirements provided.

    **Your Responsibilities:**
    1. Analyze each user request and agent response to completeness, accuracy, and relevance.
    2. Route the task to the most suitable specialist based on the current context and needs.
    3. Provide a clear explanation of your routing decision, detailing why the chosen specialist is the best fit for the next step in the workflow.
    4. Ensure that the workflow progresses efficiently towards task completion, minimizing unnecessary delays or detours.

    Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps, ultimately delivering complete and accurate solutions to the user's requests.
    """)

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"Supervisor routing decision: {goto} - Reason: {reason}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=reason, name="supervisor"
                )
            ]
        },
        goto=goto,
    )


def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer agent node that improves and clarifies user queries. Takes the original user input and transforms it into a more precise, actionalble request before passsing it back to the supervisor.
    """

    system_prompt = """
    You are a Query Refinement Specialist with expertise in transforming vague requests into clear, actionable queries. Your task is to enhance the user's original input by making it more specific, detailed, and focused, ensuring that it can be effectively addressed by the subsequent agents in the workflow.\n\n
    1. Analyze the user's original request to identify ambiguities, missing details, or areas for improvement.
    2. Resolving any ambiguities without requesting additional user.
    3. Expanding underdeveloped aspects of the request to provide a more comprehensive and actionable query.
    4. Restructuring the query to ensure clarity and precision, making it easier for the next agent to understand and act upon.
    5. Ensure that the enhanced query is clear, concise, and ready for the next stage in the workflow.

    Important: Never ask questions back to the user. Instead, make informed assumptions to fill in any gaps in the original request. Your goal is to provide a refined query that can be directly used by the next agent in the workflow.
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    enhanced_query = llm.invoke(messages)

    print(f"Enhanced query: {enhanced_query}")
    
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=enhanced_query, name="enhancer"
                )
            ]
        },
        goto="supervisor",
    )


def researcher_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Researcher agent node that gathers information using Tavily Search, Takes the current task state, performs relevant researches, and returns teh findings for validation.
    """

    research_agent = create_react_agent(
        llm,
        tools=[tavily_search],
        state_modifier="""
        You are an Information Specialist with expertise in comprehansive research. Your responsibility includes:

        1. Identifying key information needed based on the query context
        2. Gathering relevant, accurate, and up-to-date information from reliable sources.
        3. Organizing findings in a structured, easily digestible format.
        4. Citing sources when possible to establish credibility and allow for further exploration.
        5. Focusing exclusively on information gathering - avoid analysis or implementation.

        Provide thorough, factual responses without speculation or assumptions. Your goal is to deliver well-researched, actionable insights that can be used by the next agent in the workflow.
        """
    )

    result = research_agent.invoke(state)

    print(f"Research findings: {result['messages'][-1].content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="researcher"
                )
            ]
        },
        goto="validator",
    )


def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Coder agent node that generates code based on the current task state. Takes the current task state, performs relevant coding, and returns the code for review.
    """
    code_agent = create_react_agent(
        llm,
        tools=[python_repl_tool],
        state_modifier="""
        You are a Code Generation Specialist with expertise in developing efficient, functional code based on provided requirements. Your responsibilities include:
        1. Analyzing the task requirements to understand the coding needs.
        2. Writing clean, efficient code that meets the specified requirements.
        3. Ensuring the code is well-structured, documented, and easy to understand.
        4. Testing the code to ensure it functions correctly and meets the expected outcomes.
        5. Providing clear explanations of the code's functionality and how it addresses the task requirements.
        Your goal is to deliver high-quality code that can be directly used or further refined by the next agent in the workflow.
        """
    )

    result = code_agent.invoke(state)

    print(f"Generated code: {result['messages'][-1].content}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="coder"
                )
            ]
        },
        goto="validator",
    )

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Determines the next step in the workflow: 'supervisor' to escalate to the supervisor for further review, or 'FINISH' to conclude the workflow if the answer is satisfactory."
    )
    reason: str = Field(
        description="Detailed explanation of the validation decision, including whether the answer meets the requirements and why it is considered acceptable or not."
    )


def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator agent node that reviews the final answer and determines if it meets the requirements. If the answer is acceptable, it concludes the workflow; otherwise, it routes back to the supervisor for further review.
    """


    system_prompt = """
    Your task is to ensure reasonalble quality.
    Specifically, you must:
    - Review the user's question (the first message in the workflow).
    - Review the answer (the last message in the workflow).
    - If the answer addresses the core intent of the question, even if not perfectly, signal to end the workflow with 'FINISH' as the next step.
    - Only route back to the supervisor if the answer is completely off-topic, harmful, or fundamentally misunderstands the question.

    - Accept answers that are "good enough" rather than perfect
    - Prioritize workflow completing over perfect responses 
    - Give benefit of doubt to borderline answers.

    Routing Guidelines:
    1. 'supervisor' Agent: ONLY for responses that are completely incorrect or off-topic.
    2. Respond with 'FINISH' in all other cases to end the workflow gracefully.
    """

    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {user_question}"},
        {"role": "assistant", "content": f"Agent answer: {agent_answer}"}
    ]

    response = llm.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    if goto == "FINISH" or goto == "__end__" or goto == END:
        print(f"Validation complete: {goto} - Reason: {reason}")
        goto = END
    else:
        print(f"Validation escalated to supervisor: {goto} - Reason: {reason}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=reason, name="validator"
                )
            ]
        },
        goto=goto,
    )


# ----------------------- Workflow Definition -----------------------

graph = StateGraph(MessagesState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("enhancer", enhancer_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", code_node)
graph.add_node("validator", validator_node)

graph.add_edge(START, "supervisor")
app = graph.compile()
app.get_graph().print_ascii()


# ----------------------- Running the Workflow -----------------------

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Print the final response from the workflow
    final_message = result["messages"][-1]
    print(f"AI: {final_message.content}")
