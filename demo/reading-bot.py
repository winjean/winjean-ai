# 导入所需的库
import os
import gradio as gr
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class ChatbotWithRetrieval:
    def __init__(self, dir):

        # 加载Documents
        base_dir = dir # 文档的存放目录
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # 文本的分割
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        # 向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=all_splits,  # 以分块的文档
            embedding=OpenAIEmbeddings(),  # 用OpenAI的Embedding Model做嵌入
            persist_directory="e://db",  # in-memory 存储
            collection_name="my_documents")  # 指定collection_name

        # 初始化LLM
        self.llm = ChatOpenAI()

        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True
        )
        # 初始化对话历史
        self.conversation_history = ""

        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=retriever,
            memory=self.memory
        )

    def get_response(self, user_input):  # 这是为 Gradio 创建的新函数
        response = self.qa(user_input)
        # 更新对话历史
        self.conversation_history += f"你: {user_input}\nChatbot: {response['answer']}\n"
        return self.conversation_history


if __name__ == "__main__":
    folder = "e:/test"
    bot = ChatbotWithRetrieval(folder)

    # 定义 Gradio 界面
    interface = gr.Interface(
        fn=bot.get_response,  # 使用我们刚刚创建的函数
        inputs="text",  # 输入是文本
        outputs="text",  # 输出也是文本
        live=False,  # 实时更新，这样用户可以连续与模型交互
        title="智能问答",  # 界面标题
        description="请输入问题，然后点击提交。"  # 描述
    )
    interface.launch()  # 启动 Gradio 界面