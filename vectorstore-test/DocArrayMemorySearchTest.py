from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings


def main():
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["人是由恐龙进化而来", "熊猫喜欢吃天鹅肉"],
        OpenAIEmbeddings)
    retriever = vectorstore.as_retriever()
    retriever.get_relevant_documents("人从哪里来？")
    retriever.get_relevant_documents("熊猫喜欢吃什么？")
    print("")


if __name__ == '__main__':
    main()
