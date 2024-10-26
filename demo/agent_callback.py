from langchain_core.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.retrievers import BaseRetriever

load_dotenv(find_dotenv())

handler = StdOutCallbackHandler()
llm = ChatOpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")


# 构造回调：首先，在初始化链时明确设置StdOutCallbackHandler
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.invoke({"number": 2})

# 使用verbose标志：然后，让我们使用`verbose`标志来达到相同的结果
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
chain.invoke({"number": 2})

# 请求回调：最后，让我们使用请求`callbacks`来达到相同的结果
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number": 2}, {"callbacks": [handler]})

if __name__ == '__main__':
    pass
