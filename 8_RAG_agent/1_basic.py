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
def create_chroma_db(documents: List[Document], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_or_create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )

    for i, d in enumerate(documents):
        db.add(
            ids=[str(i)],
            documents=[d.page_content],
            metadatas=[d.metadata]
        )
    return db

# Create DB
db = create_chroma_db(documents=docs, path="./chroma_db", name="rag_experiment")


# -------------------- RAG Query Pipeline ------------------------

def generate_answer(user_query: str, top_k: int = 3):
    # Configure Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)

    # Embed the user query
    embed_func = GeminiEmbeddingFunction()
    query_embedding = embed_func(user_query)

    # Search for relevant documents
    results = db.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    # Extract page contents
    retrieved_docs = [result for result in results["documents"][0]]

    # Combine into context
    context = "\n\n".join(retrieved_docs)

    # Query Gemini model with the context
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are an assistant for Peak Performance Gym.
Use the following context to answer the user's question.

Context:
{context}

Question:
{user_query}

Answer:"""

    chat = model.start_chat()
    response = chat.send_message(prompt)

    return response.text


# -------------------- Example Run ------------------------

if __name__ == "__main__":
    while True:
        query = input("Ask something about the gym (type 'exit' to stop): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print("\nAI Answer:", answer)
