from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_model_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # Load environment variables from .env file

class Answer(BaseModel):
    answer: str
    confidence: float

@tool
def search_test_db(query: str, limit: int) -> str:
    """Search a small mock database and return matching results."""
    records = [
        "AI trends: AI agents are increasingly used to automate multi-step workflows.",
        "AI trends: Retrieval-augmented generation can ground answers in a document collection.",
        "AI trends: Teams are evaluating model quality, latency, and cost together.",
    ]
    matches = [record for record in records if query.casefold() in record.casefold()]
    results = matches[:max(limit, 0)]
    if not results:
        return f"No mock database results found for '{query}'."
    return "\n".join(f"{index}. {result}" for index, result in enumerate(results, 1))


@wrap_model_call
def disable_search_after_use(request, handler):
    if any(
        isinstance(message, ToolMessage) and message.name == "search_test_db"
        for message in request.state["messages"]
    ):
        request = request.override(tools=[])
    return handler(request)


def run_agent() -> None:
    """Run the LangChain course agent."""
    print("Running the LangChain course agent...")

    # Create an agent using the default LLM and tools
    # agent = create_agent(
    #     model="google_genai:gemini-3.8-flash",
    #     tools=[search_test_db],
    #     middleware=[
    #         disable_search_after_use,
    #         ToolCallLimitMiddleware(
    #             tool_name="search_test_db",
    #             run_limit=1,
    #             exit_behavior="end",
    #         )
    #     ],
    #     system_prompt=(
    #         "You are a helpful assistant. For database questions, call "
    #         "search_test_db once, then answer using its returned results. "
    #         "The tool searches mock data; do not retry the same query or claim "
    #         "that the mock data is live."
    #     ),
    #     response_format=Answer,
    # )

    local_model = ChatOpenAI(
        model="google/gemma-4-e2b",
        base_url="http://localhost:1235/v1",
        api_key="lm-studio",
    )

    agent = create_agent(
            model=local_model,
            tools=[search_test_db],
            middleware=[
                disable_search_after_use,
                ToolCallLimitMiddleware(
                    tool_name="search_test_db",
                    run_limit=1,
                    exit_behavior="end",
                )
            ],
            system_prompt=(
                "You are a helpful assistant. For database questions, call "
                "search_test_db once, then answer using its returned results. "
                "The tool searches mock data; do not retry the same query or claim "
                "that the mock data is live."
            ),
            response_format=Answer,
        )

    # Example input for the agent
    # input_text = "Summarize AI trends of 2026 till October."
    input_text2 = "Search the test database for 'AI trends' with a limit of 5 results."

    # Run the agent with the input text
    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_text2}]},
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

