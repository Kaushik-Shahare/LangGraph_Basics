from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

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
    
    human_response = interrupt(
        "Do you want to go to Node C or Node D? Type C/D"
    )
    
    if human_response.lower() == "c":

        result = Command(
            goto = "node_c",
            update = {
                "text": state["text"] + "b"
            }
        )
        return result
    elif human_response.lower() == "d":
        result = Command(
            goto = "node_d",
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

def node_d(state: State):
    print("Node D executed")
    result = Command(
        goto = END,
        update = {
            "text": state["text"] + "d"
        }
    )
    return result

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.set_entry_point("node_a")

app = graph.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

response = app.invoke({
    "text": "Kaushik"
}, config={"configurable": {"thread_id": "1"}},
    stream_mode = "updates")

# Will only run till node_b, then save the state and wait for user input
print(response)

# Now let's resume the graph from the saved state
second_response = app.invoke(Command(resume="D"), config=config, stream_mode="updates") 
print(second_response)
