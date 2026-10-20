from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # Load environment variables from .env file


def run_app() -> None:
    """Run the LangChain course application."""
    print("Running the LangChain course application...")

    information = """Ratan Naval Tata[a] (28 December 1937 – 9 October 2024) was an Indian billionaire , industrialist and philanthropist. He served as the chairman of Tata Group and Tata Sons from 1991 to 2012, and he held the position of the interim chairman from October 2016 to February 2017.[3][4] In 2000, he received the Padma Bhushan, the third-highest civilian honour in India, followed by the Padma Vibhushan, the country's second-highest civilian honour, in 2008.[5]
Ratan Tata was the son of Naval Tata, who was adopted by Ratanji Tata, son of Jamsetji Taattended the Harvard Business School Advanced Management Program in 1975.[2] He joined the Tata Group in 1962,[6] starting on the shop floor of Tata Steel. He later succeeded JRD Tata as chairman of Tata Sons upon the latter's retirement in 1991. During his tenure, the Tata Group acquired Tetley, Jaguar Land Rover, and Corus, in an attempt to turn Tata from a largely India-centric group into a global business.
Throughout his life, Tata invested in over 40 start-ups, primarily in a personal capacity, with additional investments through his firm, RNT Capital Advisors.[7][8][9]"""

    summary_template = """Given the information {information}, about a person, i want you to create
    1. A short summary of the person in 2-3 sentences.
    2. 2 interesting facts about the person in bullet points."""

    summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template
        )

    # llm = ChatOpenAI(model_name="gpt-6-luna", temperature=0.5, reasoning_effort="low", max_retries=2)

    # llm2 = ChatGoogleGenerativeAI(
    #     model="gemini-3.8-flash",
    #     temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    #     max_tokens=None,
    #     timeout=None,
    #     thinking_level="low",
    #     max_retries=2,
    # )

    # Initialize ChatOpenAI pointing to the local server
    llm3 = ChatOpenAI(
        base_url="http://localhost:1235/v1",  # Local LM Studio / Bionic server endpoint
        api_key="lm-studio",                  # Dummy key required by the client
        model="local-model",                  # Replace with your loaded model name if needed
        temperature=0.7,
    )

    

    # chain = summary_prompt_template | llm
    # response = chain.invoke({"information": information})


    # chain2 = summary_prompt_template | llm2
    # response2 = chain2.invoke({"information": information})

    # print("Response from ChatGoogleGenerativeAI:")
    # print(response2.content)
    # print(response2.content[0]['text'])

    # Invoke the local model
    chain3 = summary_prompt_template | llm3
    response3 = chain3.invoke({"information": information})
    print(response3.content)


if __name__ == "__main__":
    run_app()