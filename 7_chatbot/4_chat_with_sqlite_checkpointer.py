from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# LangGraph works with multiple threads
# Sqlite only supports one thread at a time
# So we need to create a connection to the sqlite database explicitly stating to not check for the same thread
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])],
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

# The config is used to set the thread_id for the in-memory checkpointing
# To store and use the in-memory checkpoint, you need to set the thread_id same for all the conversations
config = {"configurable": {
        "thread_id": 1,
    }
}

while True:
    user_input = input("User: ")
    if(user_input == "exit"):
        break

    result =  app.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config=config)
    print("AI: ", result["messages"][-1].content)

# The in-memory checkpointing will store the state of the conversations
# You can retrieve the state of the conversations using the get_state methos
print(app.get_state(config=config))



