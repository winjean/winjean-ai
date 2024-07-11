from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv, find_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0)
db = SQLDatabase.from_uri("mysql+pymysql://root:YY5VV5@10.20.4.75/winjean")
db_chain = SQLDatabaseChain.from_llm(llm, db)


def query_db(query):
    # print(db.get_usable_table_names())
    result = db_chain.invoke({"query": query})
    print(result["result"])


if __name__ == '__main__':
    query_db("查询表users_test中有多少条数据?")
