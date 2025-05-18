from typing import TypedDict
from langgraph.graph import END, StateGraph


class SimpleState(TypedDict):
    count: int

def increment(state: SimpleState) -> SimpleState:
    return {
        "count": state["count"] + 1
    }

def shouldContinue(state: SimpleState):
    if(state["count"] < 5):
        return "continue"
    else:
        # return END
        return "stop"

graph = StateGraph(SimpleState)

graph.add_node("increment", increment)
graph.add_conditional_edges("increment", shouldContinue, 
                            {
                                "continue": "increment",
                                "stop": END
                            }
)

graph.set_entry_point("increment")

app = graph.compile()

initial_state = {
    "count" : 0
}

result = app.invoke(initial_state)
print(result)

