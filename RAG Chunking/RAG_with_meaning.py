import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("GROQ_API_KEY")

def setup_llm_chain(word):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI expert on word separation and segmentation in the Kannada language. You will separate or segment the one long word given to you in Kannada into accurate smaller words as the word given to you doesn't have spacing. The word is taken from an ancient text called Siribhoovalaya. The word is given majorly in IAST transcription format exxcept for 'o' and 'ō' which may be in either ISO or IAST format. The resulting message that you provide should contain the list of the separated words. If you can ascertain the meanings of individual words or phrases then give the meaning of the whole compound word text in the end."),
            ("user", f"The following is the long Kannada word : {word}"),
        ]
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
    )

    return prompt | llm | StrOutputParser()


def main():
    long_Kannada_word = input("Enter the long Kannada Word-String: ")
    splitted_words = setup_llm_chain(long_Kannada_word).invoke({}).strip()
    print(splitted_words)

if __name__ == "__main__":
    main()