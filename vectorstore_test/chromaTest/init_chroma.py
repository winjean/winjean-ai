from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# from chromadb.config import Settings
# import chromadb
# from chromadb.utils import embedding_functions
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
# from transformers import AutoTokenizer, AutoModel

load_dotenv(find_dotenv())
persist_directory = "e://test/db"
embeddings = OpenAIEmbeddings()
# db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("shibing624/text2vec-base-chinese")
# model = AutoModel.from_pretrained("shibing624/text2vec-base-chinese")

"""
# client = chromadb.PersistentClient(path=persist_directory)
# embedding_function = embedding_functions.DefaultEmbeddingFunction()

# 实际创建集合
# collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
#
# documents = ["这是一个示例文档", "这是第二个示例文档"]
# ids = ["doc1", "doc2"]  # 为每个文档提供一个唯一标识符
# metadatas = [{"source": "example1"}, {"source": "example2"}]  # 可以为每个文档添加元数据
#
# # 将文档添加到集合中
# collection.add(documents=documents, ids=ids, metadatas=metadatas)

# collection = client.get_collection(name=collection_name)

# query = "查找相关文档"
# results = collection.query(query_texts=[query], n_results=2)
# print(results)

# for result in results:
    # print(f"查询结果: {result['documents'][0]}, 来源: {result['metadatas'][0]['source']}")
    # print(f"查询结果: {result['documents'][0]}")

# client = None
# with open("e://remark.txt", "r", encoding="utf-8") as f:
#     state_of_the_union = f.read()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)

# filePath = "E:/test/aa.pdf"
# loader = PyPDFLoader(file_path=filePath)
# loader = PyMuPDFLoader(file_path=filePath)

# 加载文档数据
loader = TextLoader(file_path=filePath, encoding="utf-8")  # 替换为你的文件路径
# loader = PyMuPDFLoader(file_path=filePath)
documents = loader.load()

# 分割文档为小段落
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_function = HuggingFaceEmbeddings()
# embedding_function = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
# Chroma.add_texts()

"""

collection_name = "winjean"


def save_to_chroma() -> Chroma:
    filePath = "E:/test/LangChain.pdf"
    loader = PyPDFLoader(file_path=filePath)
    # loader = PyMuPDFLoader(file_path=filePath)
    # filePath = "E:/test/aa.txt"
    # loader = TextLoader(file_path=filePath, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    try:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        # 创建检索问答链路
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

        # 查询问答系统
        query = "什么叫链?"
        response = qa_chain.invoke(query)
        print(f"问题: {query}\n答案: {response}")
    except Exception as e:
        print(f"Error connecting to Chroma: {e}")


def delete_chroma():
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    db.delete_collection()


def find_chroma():
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

    # 查询问答系统
    query = "什么是langchain链?"
    response = qa_chain.invoke(query)
    print(f"问题: {query}\n答案: {response}")

    query = "什么叫链?"
    docs = db.similarity_search(query)

    # Print results
    for doc in docs:
        print(doc.page_content)


if __name__ == '__main__':
    print("--- start save ---")
    # save_to_chroma()
    print("--- start find ---")
    # find_chroma()
    print("--- start delete ---")
    delete_chroma()
