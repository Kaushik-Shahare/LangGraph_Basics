from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State):
    print("Node A executed")
    result = Command(
        goto = "node_b",
        update = {
            "text": state["text"] + "a"
        }
    )
    return result

def node_b(state: State):
    print("Node B executed")
    result = Command(
        goto = "node_c",
        update = {
            "text": state["text"] + "b"
        }
    )
    return result

def node_c(state: State):
    print("Node C executed")
    result = Command(
        goto = END,
        update = {
            "text": state["text"] + "c"
        }
    )
    return result

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.set_entry_point("node_a")

app = graph.compile()

response = app.invoke({
    "text": "Kaushik"
})

print(response)
