from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import END, StateGraph

from nodes import reason_node, act_node
from react_state import AgentState

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT_NODE

graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.set_entry_point(REASON_NODE)

graph.add_node(ACT_NODE, act_node)

graph.add_conditional_edges(
    REASON_NODE,
    should_continue
)

graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()

result = app.invoke(
    {
        # "input": "How many days ago was the latest SpaceX launch?",
        "input": "Give me a detailed Markdown notes about queue in python in MarkDown Format. Use GeeksForGeeks for reference and w3schools for python reference. Also Give Advance Tips for best practices for queue. Also make sure to Give concise and precise notes with working examples and no Emoji in MarkDown.",
    }
)

print(result)
print("######################################################################################")
print(result["agent_outcome"].return_values["output"], "final result")
