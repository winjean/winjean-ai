from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnableBranch
from dotenv import load_dotenv, find_dotenv
import openai
import os

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
openai.base_url = os.environ['OPENAI_API_BASE']
openai.temperature = 0.7


def create_model(url="http://10.20.4.226:8000/v1") -> ChatOpenAI:
    return ChatOpenAI(model="qwen", temperature=1.0, base_url=url, api_key='test')


def conjure_function(text):
    print(text)
    return "langchain"


winjean_function = [
    {
      "name": "joke",
      "description": "你是一个知识渊博的助手。回答问题时，请保持简洁明了。",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "描述这个技术产生的背景"
          }
        },
        "required": ["question"]
      }
    }
]

qwen_chain = PromptTemplate.from_template(
    """
    你是一个知识渊博的助手。回答问题时，请保持简洁明了。
    问题: {question}
    """) | ChatOpenAI().bind(function_call={"name": "joke"}, functions=winjean_function) | JsonKeyOutputFunctionsParser(key_name="question")

langchain_chain = PromptTemplate.from_template(
    """
    你是一个知识渊博的助手。回答问题时，请保持简洁明了。
    问题: {question}
    """) | ChatOpenAI().bind(function_call={"name": "joke"}, functions=winjean_function) | StrOutputParser()

general_chain = PromptTemplate.from_template(
    """
    你是一个知识渊博的助手。回答问题时，请保持简洁明了。
    问题: {question}
    """) | ChatOpenAI().bind(stop="交流") | StrOutputParser()


def route(info):
    if "qwen" in info["question"].lower():
        return qwen_chain
    elif "langchain" in info["question"].lower():
        return langchain_chain
    else:
        return general_chain


def main():
    # model = create_model()

    # 当遇到停止词时停止回答
    # model = ChatOpenAI().bind(stop="交流")

    # template = """你是一个知识渊博的助手。回答问题时，请保持简洁明了。
    # 问题: {question}"""
    # prompt = PromptTemplate.from_template(template)
    # output = StrOutputParser()

    inputs = RunnableMap({
        "context": lambda x: x["question"],
        "question": itemgetter("question") | RunnableLambda(conjure_function)
    })

    # chain = inputs | prompt | model | output

    chain = inputs | RunnableLambda(route)

    question = "如何学好 langchain？"
    answer = chain.invoke({"question": question})
    print(f"问题: {question}\n答案: {answer}")


if __name__ == '__main__':
    main()
