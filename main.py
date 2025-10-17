# Import environment variable loader (for reading API keys from .env)
from dotenv import load_dotenv

# Import BaseModel from Pydantic — used to define structured output models
from pydantic import BaseModel

# Import OpenAI chat model wrapper from LangChain
from langchain_openai import ChatOpenAI

# Import prompt and output parser utilities
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import LangChain’s agent tools and executor
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Import your custom tools (assumed defined in tools.py)
from tools import search_tool, wiki_tool, save_tool


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
# This reads API keys or config values stored in a .env file.
# Make sure your .env file includes your OpenAI key:
# OPENAI_API_KEY=sk-xxxxxxxxxxxx
load_dotenv()


# ---------------------------------------------------------
# 2. Define the output structure for the model’s response
# ---------------------------------------------------------
# Using Pydantic BaseModel to enforce a consistent format.
class ResearchResponse(BaseModel):
    topic: str         # The main research topic
    summary: str       # A concise summary of findings
    sources: list[str] # List of sources or URLs used
    tools_used: list[str] # List of tools (search, wiki, etc.) the agent used


# ---------------------------------------------------------
# 3. Initialize the language model (GPT)
# ---------------------------------------------------------
# ChatOpenAI lets LangChain use OpenAI's GPT models.
# - "gpt-4o-mini" is a fast, cost-efficient model.
# - temperature=0 ensures deterministic (consistent) answers.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=None)


# ---------------------------------------------------------
# 4. Create a parser to structure the model’s output
# ---------------------------------------------------------
# The PydanticOutputParser converts the raw text output
# into a validated ResearchResponse object.
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# ---------------------------------------------------------
# 5. Create the system prompt template
# ---------------------------------------------------------
# This defines *how* the model should behave.
# It uses placeholders for chat history, user query, and agent steps.
prompt = ChatPromptTemplate.from_messages(
    [
        # System role: sets the model's behavior
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text:
            {format_instructions}
            """,
        ),
        # Placeholder for conversation history (if any)
        ("placeholder", "{chat_history}"),
        # The actual user query
        ("human", "{query}"),
        # Placeholder for tool call outputs and reasoning chain
        ("placeholder", "{agent_scratchpad}"),
    ]
# Insert the structured output format automatically
).partial(format_instructions=parser.get_format_instructions())


# ---------------------------------------------------------
# 6. Define available tools for the agent
# ---------------------------------------------------------
# These are helper functions (LangChain Tools) that the model
# can call automatically when it needs to search or save information.
tools = [search_tool, wiki_tool, save_tool]


# ---------------------------------------------------------
# 7. Create the tool-calling agent
# ---------------------------------------------------------
# The agent can interpret prompts, decide when to use tools,
# and combine results into a coherent response.
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


# ---------------------------------------------------------
# 8. Wrap the agent in an executor
# ---------------------------------------------------------
# AgentExecutor manages the full agent workflow:
# - passing inputs and outputs
# - invoking tools
# - returning structured results
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ---------------------------------------------------------
# 9. Main program: take user input and run the agent
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ask user what topic they want to research
    query = input("What can I help you research? ")

    # Invoke the agent with the user’s query
    raw_response = agent_executor.invoke({"query": query})

    # -----------------------------------------------------
    # 10. Parse and print the structured result
    # -----------------------------------------------------
    try:
        # Depending on LangChain version, output can be list or string.
        text_output = (
            raw_response.get("output")[0]["text"]
            if isinstance(raw_response.get("output"), list)
            else raw_response.get("output")
        )

        # Convert raw text into the structured Pydantic model
        structured_response = parser.parse(text_output)

        # Print the parsed object (pretty formatted)
        print(structured_response)

    except Exception as e:
        # Handle any parsing errors and print raw output for debugging
        print("Error parsing response:", e)
        print("Raw Response:", raw_response)
