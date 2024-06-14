from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import os

os.environ["OPENAI_API_KEY"] = "test"

with open("e://remark.txt", "r", encoding="utf-8") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings,
                              metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
                              persist_directory="e://db")

docsearch.persist()
docsearch = None

if __name__ == '__main__':
    pass
