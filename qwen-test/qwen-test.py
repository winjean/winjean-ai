from langchain_community.llms import Qwen
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = Qwen(temperature=0.7)

template = "Question: {question}"
prompt = PromptTemplate(template=template, input_variables=["question"])
qa_chain = LLMChain(llm=llm, prompt=prompt)

question = "什么是量子计算？"
answer = qa_chain.run(question)
print(answer)

if __name__ == '__main__':
    pass
