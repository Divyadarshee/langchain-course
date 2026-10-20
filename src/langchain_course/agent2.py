from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_model_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file

class Answer(BaseModel):
    answer: str
    confidence: float


def search(query: str):
    """
    Tool that searches over internet
    
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """

    print(f"Searching for: {query}")
    return "Tokyo weather is sunny, 25°C."

def calculate(query: str):
    """
    Tool that performs calculations
    
    Args:
        query (str): The calculation query.
    Returns:
        str: The calculation results.
    """

    print(f"Calculating: {query}")
    return "The result of the calculation is 42."

def decide(query: str):
    """
    Tool that makes decisions based on the query
    
    Args:
        query (str): The decision-making query.
    Returns:
        str: The decision results.
    """

    print(f"Deciding on: {query}")
    return "Based on the analysis, the best course of action is to proceed with option A."

def run_agent2() -> None:
    """Run the LangChain course agent."""
    print("Running the LangChain course agent...")

    llm_local = ChatOpenAI(
        model="google/gemma-4-e2b",
        base_url="http://localhost:1235/v1",
        api_key="lm-studio",
    )

    # agent = create_agent(
    #         model=llm_local,
    #         tools=[search, calculate, decide],
    #         system_prompt=(
    #             "You are a helpful assistant. For any questions, call "
    #             "the appropriate tool, then answer using its returned results."
    #         ),
    #         response_format=Answer,
    #     )

    llm_google = ChatGoogleGenAI(
        model="gemini-3.8-flash",
        temperature=0,
        max_output_tokens=512,
    )

    agent = create_agent(
            # model="google_genai:gemini-3.8-flash",
            model = llm_google,
            tools=[search, calculate, decide],
            system_prompt=(
                "You are a helpful assistant. For any questions, call "
                "the appropriate tool, then answer using its returned results."
            ),
            response_format=Answer,
        )

    # Example input for the agent
    # input_text = "Summarize AI trends of 2026 till October."
    # input_text2 = "How is the weather in Tokyo today?"
    input_text3 = "Calculate the sum of 15 and 28. and also decide whether to invest in stocks or bonds based on the current market trends."

    # Run the agent with the input text
    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_text3}]},
        config={"recursion_limit": 8},
    )

    print("Agent response:")
    print(response)
    structured_response = response.get("structured_response")
    if structured_response is not None:
        print(f"Structured response: {structured_response}")
    else:
        print("The agent stopped before producing a structured response:")
        print(response["messages"][-1].content)

