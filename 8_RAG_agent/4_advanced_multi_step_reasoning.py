import os
from dotenv import load_dotenv
from langchain.schema import Document
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List

# Load .env variables
load_dotenv()

# --------------------- Sample Documents -------------------------
docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"}
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"}
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"}
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"}
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"}
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"}
    )
]



# ---------------------- Embedding Function ----------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")

        genai.configure(api_key=gemini_api_key)

        # Convert input to string (if input is a list of strings or a single string)
        if isinstance(input, list):
            input_text = " ".join(input)
        else:
            input_text = input

        # Get the embedding from Gemini
        response = genai.embed_content(
            model="models/embedding-001",
            content=input_text,
            task_type="retrieval_document",
            title="Custom query"
        )

        embedding = response["embedding"]

        # If it's a NumPy array, convert it to a plain Python list
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        # Return a **single-level list of lists**, as expected by ChromaDB
        return [embedding]

# ------------------------ Create ChromaDB -----------------------
# def create_chroma_db(documents: List[Document], path: str, name: str):
    # chroma_client = chromadb.PersistentClient(path=path)
    # db = chroma_client.get_or_create_collection(
        # name=name,
        # embedding_function=GeminiEmbeddingFunction()
    # )
 
    # for i, d in enumerate(documents):
        # db.add(
            # ids=[str(i)],
            # documents=[d.page_content],
            # metadatas=[d.metadata]
        # )
    # return db

# Create DB
# db = create_chroma_db(documents=docs, path="./chroma_db", name="rag_experiment")
# retriever = db.as_retriever(search_type="mmr", search_kwargs = {"k": 4})

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create Chroma vectorstore
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# Now you can use as_retriever()
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})


# -------------------------- RAG Chain --------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

template = """
Answer the question based on the following context and the ChatHistory. Especially take the latest messages into account.

ChatHistory: {history}

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm


# -------------------- Graph Nodes --------------------
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# Agent State that holds the conversation history, documents, and other relevant information
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int # max 3 rephrases
    question: HumanMessage


# GradeQuestion model is to classify whether the question is about the specified topics
class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? IF yes -> 'Yes', if not -> 'No'"
    )


# Function to rewrite the question based on the context and chat history
def question_rewriter(state: AgentState) -> AgentState:
    print("Rewriting question...")

    # Reset state variables except for the 'question' and 'messages'
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1] # Exclude the last question message
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are an expert at rewriting questions based on context and chat history."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        rephrased_response = llm.invoke(rephrase_prompt.format_messages())
        better_question = rephrased_response.content.strip()
        print(f"Rephrased question: {better_question}")
        state["rephrased_question"] = better_question

    else:
        print("No previous messages to rephrase the question.")
        state["rephrased_question"] = state["question"].content

    return state


# Function to classify the question
def question_classifier(state: AgentState) -> AgentState:
    print("Classifying question...")
    system_message = SystemMessage(
        content="""
        You are a classifier that determines whether a user's question is about one of the following topics:

        1. Gym History & Founder
        2. Operating Hours
        3. Membership Plans
        4. Fitness Classes
        5. Personal Trainers
        6. Facilities & Equipment
        7. Anything else about any of these topics, response with 'Yes'. Otherwise, respond with 'No'.
        """
    )

    human_message = HumanMessage(
        content=f"User question: {state["rephrased_question"]}"
    )

    # Create the prompt for classification

    # The prompt is structured to include the system message and the user's question
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Create a structured LLM that will return the classification result
    structured_llm = llm.with_structured_output(GradeQuestion)
    # Invoke the LLM to classify the question
    grader_llm = grade_prompt | structured_llm

    # Invoke the LLM with the current state
    result = grader_llm.invoke({})

    state["on_topic"] = result.score.strip()
    print(f"Classifier result: {state['on_topic']}")
    return state


# Function to route based on on_topic_router
def on_topic_router(state: AgentState):
    print("Routing based on topic...")
    on_topic = state["on_topic"].lower()
    if on_topic == "yes":
        print("Routing to RAG generation...")
        return "retrieve"
    else:
        print("Routing to off_topic response...")
        return "off_topic_response"


# Retrieve relevant documents based on the rephrased question
def retrieve(state: AgentState) -> AgentState:
    print("Retrieving relevent documents...")
    documents = retriever.invoke(
        state["rephrased_question"]
    )
    print(f"retrieve: Retrieved {len(documents)} documents")
    state["documents"] = documents
    return state


# GradeDocument model is to classify whether the document is relevant to the question
class GradeDocument(BaseModel):
    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes', if not -> 'No'"
    )


# Function to grade the retrieved documents based on their relevance to the question
def retrieval_grader(state: AgentState) -> AgentState:
    print("Grading retrieved documents...")
    system_message = SystemMessage(
        content="""
        You are a assessing the relevance of a retrieved document to a user question.
        Only answer with 'Yes' if the document is relevant to the question, otherwise answer with 'No'.
        """
    )

    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"user question: {state['rephrased_question']}\n\nRetrieved document: \n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})

        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)
            print(f"Document '{doc.metadata['source']}' is relevant to the question.")

    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    print(f"relevant_docs: {len(relevant_docs)} documents are relevant to the question.")
    return state


# proceed_router function to determine the next step based on document relevance
def proceed_router(state: AgentState):
    print("Routing based on document relevance...")
    if state["proceed_to_generate"]:
        print("Proceeding to answer generation...")
        return "generate_answer"
    elif state["rephrase_count"] >= 2:
        print("Maximum rephrase attempts reached, routing to off_topic_response...")
        return "cannot_answer"
    else:
        print("Routing to refine_qestion...")
        return "refine_question"


# refine_question function to refine the rephrased question
def refine_question(state: AgentState) -> AgentState:
    print("Refining question...")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached, cannot refine further.")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="You are a helpful assessant that helps refine user questions to make them more specific and answerable."
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined version of this question that is more specific and answerable."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    response = llm.invoke(refine_prompt.format_messages())
    refined_question = response.content.strip()
    print(f"Refined question: {refined_question}")
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state


# generate_answer function to generate the final answer based on the rephrased question and retrieved documents
def generate_answer(state: AgentState) -> AgentState:
    print("Generating answer...")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must contain 'messages' key with a list of messages.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {
            "history": history,
            "context": "\n\n".join([doc.page_content for doc in documents]),
            "question": rephrased_question
        }
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"Generated answer: {generation}")
    return state


def cannot_answer(state: AgentState) -> AgentState:
    print("Cannot answer the question, providing a default response.")
    if not "messages" in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry, I cannot answer that question."))
    return state


def off_topic_response(state: AgentState) -> AgentState:
    print("Off-topic response.")
    if not "messages" in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question as it does not seem to be related to Peak Performance Gym."))
    return state


# -------------------- Build & Compile the Graph --------------------
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

workflow = StateGraph(AgentState)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("cannot_answer", cannot_answer)
workflow.add_node("refine_question", refine_question)

workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve": "retrieve",
        "off_topic_response": "off_topic_response"
    }
)
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer"
    }
)
workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("question_rewriter")
graph = workflow.compile(checkpointer=checkpointer)
# graph = workflow.compile()

config = {"config": {
        "thread_id": 1,
    }
}

# Print the graph structure
print("Graph structure:")
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


# -------------------- Example Run --------------------
# input_data = {"question": HumanMessage(content="What dos the company Apple do?")}
# print(graph.invoke(input_data, config={"thread_id": 1}))
# 
# input_data = {
#     "question": HumanMessage(
#         content="What is the cancelation policy for Peak Performance Gym membership?"
#     )
# }
# print(graph.invoke(input_data, config={"thread_id": 1}))
# 
# 
# input_data = {
#     "question": HumanMessage(
#         content="Who founded Peak Performance Gym and when?"
#     )
# }
# print(graph.invoke(input_data, config={"thread_id": 1}))


input_data = {
    "question": HumanMessage(
        content="What are the timings of the gym?"
    )
}
print(graph.invoke(input_data, config={"thread_id": 1}))


input_data = {
    "question": HumanMessage(
        content="What about the weekends?"
    )
}
print(graph.invoke(input_data, config={"thread_id": 1}))
