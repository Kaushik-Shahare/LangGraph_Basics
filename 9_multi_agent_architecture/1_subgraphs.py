from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

load_dotenv()

# ChildState is used to define the state of the child node
class ChildState(TypedDict):
    messages: Annotated[list, add_messages]


search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")

llm_with_tools = llm.bind_tools(tools)

def agent(state: ChildState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


def tools_router(state: ChildState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        # If the last message has tool calls, we will invoke the tool node
        return "tool_node"
    else:
        # If the last message does not have tool calls, we will invoke the agent node
        return END

tool_node = ToolNode(tools=tools)

subgraph = StateGraph(ChildState)

subgraph.add_node("agent", agent)
subgraph.add_node("tool_node", tool_node)
subgraph.set_entry_point("agent")

subgraph.add_conditional_edges("agent", tools_router)
subgraph.add_edge("tool_node", "agent")

search_app = subgraph.compile()
search_app.get_graph().print_ascii()


# response = search_app.invoke({
    # "messages": [HumanMessage(content="What is LangGraph?")]
# })
# print(response["messages"][-1].content)




# ----------------------------------- Case 1 : Shared Schema -----------------------------------

class ParentState(TypedDict):
    messages: Annotated[list, add_messages]


parent_graph = StateGraph(ParentState)

# Add the subgraph to the parent graph as a node
parent_graph.add_node("search_agent", search_app)

parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)

parent_app = parent_graph.compile()
parent_app.get_graph().print_ascii()


response = parent_app.invoke({
    "messages": [HumanMessage(content="What is LangGraph?")]
})
print(response["messages"][-1].content)



# ----------------------------------- Case 2 : Different Schema -----------------------------------

class ParentState2(TypedDict):
    query: str
    response: str


def search_agent(state: ParentState2):
    # Transform from parent schema to subgraph schema

    subgraph_input = {
        "messages": [HumanMessage(content=state["query"])]
    }

    # Invoke the subgraph
    subgraph_result = search_app.invoke(subgraph_input)

    # Transform response back to parent schema
    assistant_message = subgraph_result["messages"][-1]
    return {"response": assistant_message.content}

parent_graph = StateGraph(ParentState2)

parent_graph.add_node("search_agent", search_agent)

parent_graph.add_edge(START, "search_agent")
parent_graph.add_edge("search_agent", END)

parent_app = parent_graph.compile()
parent_app.get_graph().print_ascii()

response = parent_app.invoke({
    "query": "What is LangGraph?"
})
print(response["response"])


