from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama


def main():
    from langchain.prompts import PromptTemplate

    ollama_llm = Ollama(base_url="http://localhost:11434", model="qwen:14b")

    template = """你是一个知识渊博的助手。回答问题时，请保持简洁明了。
        问题: {question}"""
    prompt = PromptTemplate.from_template(template)

    # 使用PromptTemplate和Ollama LLM创建一个Chain
    output = StrOutputParser()
    chain = prompt | ollama_llm | output

    # 调用Chain获取回答
    question = "如何学好 langchain？"
    response = chain.invoke({"question": question})
    print(response)


if __name__ == '__main__':
    main()
