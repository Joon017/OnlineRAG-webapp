from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access variables using os.getenv
api_key = os.getenv("API_KEY")

# Use the variables as needed in your application
print(f"API Key: {api_key}")

llm = ChatOpenAI(temperature=0.0, openai_api_key=api_key, model_name="gpt-3.5-turbo-16k")
# result = llm.predict("Hello, how are you?")
# print(result)

query = "What is the capital of France?"
sources_combined = "Lee Hsien Loong is the prime minister of Singapore"

system_template = ("You are a helpful assistant that considers the context provided and provide an objective answer to the user's question. If the provided context is not relevant or does not contain the answer, state so.")
human_template = """Given the question: {query},
                    Answer based on the given context. If no relevant information in the context, state so: {sources_combined}
                """

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

answer = llm(chat_prompt.format_prompt(query=query, sources_combined=sources_combined).to_messages())
print(answer.content)







