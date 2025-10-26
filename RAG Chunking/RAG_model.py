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

def setup_llm_chain(word, chakra_no):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are an AI expert on word separation and segmentation in the Kannada language. You will separate or segment the one long word given to you in Kannada into accurate smaller words as the word given to you doesn't have spacing. The word is taken from an ancient text called Siribhoovalaya. The word is given majorly in IAST transcription format except for 'o' and 'ō' which may be in either ISO or IAST format. The resulting message that you provide should contain the list of the separated words only in IAST format. Example for response format is: ch1_1 = [\"asta\", \"fgtd\", \"agdsau\", ...]. Here, ch1_1 is the text given as {chakra_no}. Be sure to use straight quotes and not curly quotes."),
            ("user", f"The following is the long Kannada word : {word}"),
        ]
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=groq_api_key,
    )

    return prompt | llm | StrOutputParser()


def main():
    chakra_no = input("Enter the chakra number: ")
    long_Kannada_word = input("Enter the long Kannada Word-String: ")
    splitted_words = setup_llm_chain(long_Kannada_word, chakra_no).invoke({}).strip()
    print(splitted_words)
    with open(f"../IAST-Chakras/IAST_chakra_{chakra_no}.py", 'w', encoding='utf-8') as f:
        f.write(splitted_words)

if __name__ == "__main__":
    main()