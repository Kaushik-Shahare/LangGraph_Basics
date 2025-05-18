from typing import TypedDict, List, Annotated
from langgraph.graph import END, StateGraph
import operator


class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    newCount = state["count"] + 1
    return {
        "count": newCount,
        "sum": newCount,
        "history": [newCount]
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
    "count" : 0,
    "sum" : 0,
    "history": []
}

result = app.invoke(initial_state)
print(result)

