import os
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain

os.environ["OPENAI_API_KEY"] = "test"

embeddings = OpenAIEmbeddings()

docsearch = Chroma(
    persist_directory="e://test/db",
    embedding_function=embeddings,
    collection_name="winjean",
)

a = docsearch.similarity_search("What's your name?", k=1)
print(a)

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())


user_input = input("What's your question: ")

result = chain.invoke({"question": user_input}, return_only_outputs=True)

print("Answer: " + result["answer"].replace('\n', ' '))
print("Source: " + result["sources"])

if __name__ == '__main__':
    pass
