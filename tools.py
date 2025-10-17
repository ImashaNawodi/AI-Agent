# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
# LangChain’s community tools provide built-in integrations for
# Wikipedia and DuckDuckGo (for web searches).
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

# The Tool class allows you to define custom functions that the agent can call.
from langchain.tools import Tool

# Used to add timestamps when saving results to a file.
from datetime import datetime


# ---------------------------------------------------------
# 1. Custom Save Function
# ---------------------------------------------------------
def save_to_txt(data: str, filename: str = "research_output.txt"):
    """
    Save research results into a text file with a timestamp.

    Parameters:
    - data (str): The text or structured data you want to save.
    - filename (str): The file name to append the data to. Default is 'research_output.txt'.

    Returns:
    - str: A confirmation message showing where the data was saved.
    """

    # Create a readable timestamp for the saved record
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format the text to make each entry clearly separated
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Open the file in append mode and write the formatted text
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    # Return success message to the agent
    return f"Data successfully saved to {filename}"


# ---------------------------------------------------------
# 2. Tool: Save Tool Wrapper
# ---------------------------------------------------------
# Wrap the above save function into a LangChain Tool.
# The agent can automatically call this when it decides
# to store research data locally.
save_tool = Tool(
    name="save_text_to_file",          # Unique identifier for the tool
    func=save_to_txt,                  # Function it will execute
    description="Saves structured research data to a text file.",  # Description shown to the model
)


# ---------------------------------------------------------
# 3. Tool: DuckDuckGo Search
# ---------------------------------------------------------
# This allows the agent to perform real-time web searches
# for any topic using DuckDuckGo.
search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",                     # Tool name (used in prompt)
    func=search.run,                   # The callable function
    description="Search the web for information",  # What this tool does
)


# ---------------------------------------------------------
# 4. Tool: Wikipedia Lookup
# ---------------------------------------------------------
# The WikipediaAPIWrapper fetches relevant text snippets
# from Wikipedia articles. You can control:
# - top_k_results: number of search hits
# - doc_content_chars_max: how many characters per result to include
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

# WikipediaQueryRun is a wrapper that formats results nicely
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# ---------------------------------------------------------
# ✅ Summary
# ---------------------------------------------------------
# This file defines 3 tools that your research agent can use:
# 1. search_tool   → Web search (DuckDuckGo)
# 2. wiki_tool     → Wikipedia data lookup
# 3. save_tool     → Save structured research results to a local file
#
# The main agent (in your other script) imports these tools and
# decides automatically when to use them during a research task.
