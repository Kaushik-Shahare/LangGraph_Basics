from typeing_extensions import Annotated, TypedDict
from typing import Optional
from pydantic_SO import llm

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    # Annotated[type, default_value, description]
    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as

    # setup: str                      # no default, no description
    # setup: Annotated[str, ...]      # no default, no description
    # setup: Annotated[str, "foo"]    # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke."]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)

response = structured_llm.invoke("Tell me a joke about cats.")
print(response)
