from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set up the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Define the structured output model
class Country(BaseModel):
    """Information about a country"""
    name: str = Field(description="Name of the country")
    language: str = Field(description="Official language of the country")
    capital: str = Field(description="Capital city of the country")

# Link the model with structured output
structured_llm = llm.with_structured_output(Country)

# Invoke the LLM with a prompt
result = structured_llm.invoke("Tell me about France")

# Print the result
print(result)
