from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import openai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
openai.base_url = os.environ['OPENAI_API_BASE']
openai.temperature = 0.7

filePath = "E:/AI/6147.pdf"


def read_pdf():
    loader = PyPDFLoader(file_path=filePath, extract_images=False)
    documents = loader.load()
    for doc in documents:
        print(doc.page_content)


def ask_pdf():
    loader = PyPDFLoader(file_path=filePath)
    documents = loader.load()
    chain = load_qa_chain(ChatOpenAI())
    answer = chain.invoke(input={"input_documents": documents[3:3], "question": "这个文档一共有多少个中文汉字和多少个英文单词？"})
    print(f"问题: {answer['question']}\n答案: {answer['output_text']}")


def main():
    # read_pdf()
    ask_pdf()


if __name__ == '__main__':
    main()
