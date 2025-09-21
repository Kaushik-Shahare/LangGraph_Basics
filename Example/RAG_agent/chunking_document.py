from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

text = """
This is a long document that needs to be split into smaller chunks for processing. The document contains multiple paragraphs, sections, and various types of content. The goal is to ensure that each chunk is manageable in size while preserving the context and meaning of the original text.
"""

# Initialize the text splitter using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

# Split the text into chunks
documents = text_splitter.create_documents([text])

# Print the resulting document chunks
for i, doc in enumerate(documents):
    print(f"--- Document Chunk {i+1} ---")
    print(doc.page_content)
    print()


#########################################################################
# Alternative: Using CharacterTextSplitter
#########################################################################

# Initialize the text splitter using CharacterTextSplitter
char_text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

# Split the text into chunks
documents_char = char_text_splitter.create_documents([text])

# Print the resulting document chunks
for i, doc in enumerate(documents_char):
    print(f"--- Character Document Chunk {i+1} ---")
    print(doc.page_content)
    print()



#########################################################################
# Alternative: Using RecursiveCharacterTextSplitter for specific programming languages
#########################################################################

# Example for Python code
code_text = """
def example_function():
    print("This is an example function.")
    for i in range(10):
        print(i)
"""

# Initialize the text splitter for Python code
code_text_splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=50,
    chunk_overlap=10,
)

# Split the code text into chunks
documents_code = code_text_splitter.create_documents([code_text])
# Print the resulting code document chunks
for i, doc in enumerate(documents_code):
    print(f"--- Code Document Chunk {i+1} ---")
    print(doc.page_content)
    print()


#########################################################################
# Semantic Chunking Example
#########################################################################

from langchain_experimental.text_splitter import SemanticChunker
# gemini Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # can also be "standard_deviation" or "interquartile"
)

documents_semantic = text_splitter.create_documents([text])
# Print the resulting semantic document chunks
for i, doc in enumerate(documents_semantic):
    print(f"--- Semantic Document Chunk {i+1} ---")
    print(doc.page_content)
    print()



#########################################################################
# Agentic Chunking Example
# Proportion based chunking using an LLM to determine chunk boundaries
#########################################################################

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub

prompt_template = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI(model = "gpt-4o", temperature=0)
runnable = prompt_template | llm

class Sentences(BaseModel):
    sentences: List[str]

extraction_chain = create_extraction_chain_pydantic(
        pydantic_schema= Sentences,
        llm=llm,
)

def get_propositions(text):
    runnable_output = runnable.invoke({"text": text}).content
    propositions = extraction_chain.predict_and_parse(runnable_output)["text"][0].sentences
    return propositions

paragraphs = text.split("\n\n")

text_propositions = []
for i, para in enumerate(paragraphs):
    props = get_propositions(para)
    text_propositions.extend(props)
    print(f"--- Paragraph {i+1} Propositions ---")


# grouping chunk

# from agentic_chunker import AgenticChunker
# ac = AgenticChunker()
# ac.add_propositions(text_propositions)
# print(ac.pretty_print_chunks())
# chunks = ac.get_chunks(get_type='list_of_strings')
# print(chunks)
# documents = [Document(page_content=chunk) for chunk in chunks]
# rag(documents, "agentic-chunking")
