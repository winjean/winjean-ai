from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from flask import Flask, request, render_template
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)


def get_question_answer(question: str) -> str:
    logging.info(f"question: {question}")
    # ollama_llm = Ollama(base_url="http://localhost:11434", model="qwen:14b")
    ollama_llm = Ollama(base_url="http://10.20.7.103:11434", model="qwen2:7b")

    template = """你是一个知识渊博的助手。回答问题时，请保持简洁明了。
        问题: {question}"""
    prompt = PromptTemplate.from_template(template)

    # 使用PromptTemplate和Ollama LLM创建一个Chain
    output = StrOutputParser()
    chain = prompt | ollama_llm | output

    # 调用Chain获取回答
    # question = "如何学好 langchain？"
    return chain.invoke({"question": question})


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        # 接收用户输入作为问题
        question = request.form.get('question')
        result = get_question_answer(question)

        # 把大模型的回答结果返回网页进行渲染
        return render_template('index.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
