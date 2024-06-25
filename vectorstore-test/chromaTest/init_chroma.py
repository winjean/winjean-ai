from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from chromadb.config import Settings
import chromadb
from chromadb.utils import embedding_functions


os.environ["OPENAI_API_KEY"] = "test"
persist_directory = "e://db"
client = chromadb.PersistentClient(path=persist_directory)

collection_name = "my_documents2"
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

collection = client.get_collection(name=collection_name)

query = "查找相关文档"
results = collection.query(query_texts=[query], n_results=2)
print(results)


# for result in results:
    # print(f"查询结果: {result['documents'][0]}, 来源: {result['metadatas'][0]['source']}")
    # print(f"查询结果: {result['documents'][0]}")

client = None
# with open("e://remark.txt", "r", encoding="utf-8") as f:
#     state_of_the_union = f.read()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)

# filePath = "E:/AI/6147.pdf"

# loader = PyPDFLoader(file_path=filePath)
# documents = loader.load()

# embeddings = OpenAIEmbeddings()
# Chroma.add_texts()

# 加载文档数据
# filePath = "E:/remark.txt"
# loader = TextLoader(filePath, encoding="utf-8")  # 替换为你的文件路径
# documents = loader.load()

# 分割文档为小段落
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)


# docsearch = Chroma.add_documents(documents=documents, embedding=embeddings, persist_directory="e://db")

# metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
# db = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     collection_name="winjean",
#     persist_directory="e://db")
# db.persist()
# db = None

if __name__ == '__main__':
    pass
