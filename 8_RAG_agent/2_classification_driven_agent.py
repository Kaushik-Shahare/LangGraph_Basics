import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# ---------- 1. Setup Environment ----------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ---------- 2. Gemini Embedding Function ----------

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, list):
            input_text = " ".join(input)
        else:
            input_text = input

        response = genai.embed_content(
            model="models/embedding-001",
            content=input_text,
            task_type="retrieval_document",
            title="Custom query"
        )
        embedding = response["embedding"]
        return [embedding]  # Ensure it's a list of lists


# ---------- 3. Knowledge Base ----------

docs = [
    Document(page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen.", metadata={"source": "about.txt"}),
    Document(page_content="The gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends: 7:00 AM to 9:00 PM.", metadata={"source": "hours.txt"}),
    Document(page_content="Basic (₹1,500), Standard (₹2,500), Premium (₹4,000) membership plans are available.", metadata={"source": "membership.txt"}),
    Document(page_content="Classes include Yoga, HIIT, Zumba, and CrossFit, held throughout the week.", metadata={"source": "classes.txt"}),
    Document(page_content="Our head trainer, Neha Kapoor, specializes in rehabilitation fitness.", metadata={"source": "trainers.txt"}),
    Document(page_content="Facilities: Cardio zone, spin studio, pool, sauna, and juice bar.", metadata={"source": "facilities.txt"}),
]

# ---------- 4. Setup ChromaDB ----------

db = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    name="rag_experiment",
    embedding_function=GeminiEmbeddingFunction()
)

# Add docs (only if empty, to avoid duplicates)
if not db.peek()["ids"]:
    for i, d in enumerate(docs):
        db.add(ids=[str(i)], documents=[d.page_content], metadatas=[d.metadata])

retriever = lambda query: db.query(query_embeddings=GeminiEmbeddingFunction()(query), n_results=3)["documents"][0]


# ---------- 5. RAG Generation ----------

def generate_answer_from_context(context: str, user_query: str):
    prompt = f"""You are an assistant for Peak Performance Gym.
Use the following context to answer the user's question.

Context:
{context}

Question:
{user_query}

Answer:"""
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    chat = model.start_chat()
    return chat.send_message(prompt).text


# ---------- 6. Agent State ----------

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str


# ---------- 7. Classifier Model ----------

class GradeQuestion(BaseModel):
    score: str = Field(description="Yes or No if the question is about Peak Performance Gym.")

def classify_question(question: str) -> str:
    system_prompt = """You are a classifier for Peak Performance Gym-related topics:
    
- Gym History & Founder
- Operating Hours
- Membership Plans
- Classes
- Trainers
- Facilities

If the question is about these, respond with 'Yes', else 'No'."""

    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()
    prompt = f"{system_prompt}\n\nUser question: {question}\n\nAnswer:"
    response = chat.send_message(prompt)
    return response.text.strip()


# ---------- 8. Graph Nodes ----------

def classify_node(state: AgentState):
    print("--- CLASSIFYING ---")
    question = state["messages"][-1].content
    state["on_topic"] = classify_question(question)
    print(f"Classifier result: {state['on_topic']}")
    return state

def retrieve_node(state: AgentState):
    print("--- RETRIEVING DOCS ---")
    question = state["messages"][-1].content
    docs_content = retriever(question)
    state["documents"] = [Document(page_content=doc) for doc in docs_content]
    return state

def generate_answer_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    question = state["messages"][-1].content
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    answer = generate_answer_from_context(context, question)
    state["messages"].append(AIMessage(content=answer))
    return state

def off_topic_node(state: AgentState):
    print("--- OFF-TOPIC RESPONSE ---")
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question as it does not seem to be related to Peak Performance Gym."))
    return state


# ---------- 9. Router ----------

def route_on_topic(state: AgentState):
    print(f"--- ROUTING: {state['on_topic']} ---")
    return "on_topic" if state["on_topic"].lower() == "yes" else "off_topic"


# ---------- 10. Build & Compile the Graph ----------

workflow = StateGraph(AgentState)

workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("off_topic", off_topic_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_on_topic, {"on_topic": "retrieve", "off_topic": "off_topic"})
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic", END)

graph = workflow.compile()


# ---------- 11. Example Run ----------

if __name__ == "__main__":
    while True:
        query = input("Ask something about the gym (type 'exit' to stop): ")
        if query.lower() == "exit":
            break
        state = graph.invoke({"messages": [HumanMessage(content=query)]})
        print("\nAI Answer:", state["messages"][-1].content)
