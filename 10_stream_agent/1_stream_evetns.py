from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_tool = TavilySearchResults(search_depth="basic")
tools = [search_tool]

llm_with_tools = llm.bind_tools(tools)

def model(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        # If the last message has tool calls, we will invoke the tool node
        return "tool_node"
    else:
        # If the last message does not have tool calls, we will invoke the model node
        return END


tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)

graph.add_node("model", model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")

graph.add_conditional_edges("model", tools_router)
graph.add_edge("tool_node", "model")

app = graph.compile()

app.get_graph().print_ascii()

input = {
    "messages": ["What's the weather like in San Francisco?"]
}

events = app.stream(input=input, stream_mode="values")

for event in events:
    print(event)
