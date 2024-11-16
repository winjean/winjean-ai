import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4-flash",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)

word_phrase = """
你是日志精灵，你专注于处理用户提供的日志数据，通过智能解析生成固定格式的解码器。
你的能力有:
    - 自动识别日志格式
    - 快速生成解码器
    - 支持多种日志类型解析

提供如下格式化输出:
decoder:
    parent: useradd
    name: useradd-newuser
    conditions:
        - regex:
            field: message
            pattern:'new user'
    processors:
        - regex:
            field: message
            offset: whole
            pattern: 'new user:\s+name=(\S+)\.*'
            targets:['name']
"""

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(word_phrase),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
ans =conversation.invoke({"question":
"""
www.pipixia.org---100.122.17.191 - - [06/Oct/2024:03:46:21 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white,green,blue,red&sort=_default HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=black%2Cwhite%2Cgreen%2Cblue%2Cred" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.130.32"
www.pipixia.org---100.122.17.142 - - [06/Oct/2024:03:48:25 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=blue%2Cblack%2Cwhite" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.151.174"
"""})


# Nov 10 00:11:31 localhost sshd[528649]: pam_unix(sshd:auth): check pass; user unknown
# Nov 10 00:11:31 localhost sshd[528649]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=139.162.133.194
# Nov 10 00:11:33 localhost sshd[528649]: Failed password for invalid user changcan from 139.162.133.194 port 42046 ssh2
# Nov 10 00:11:35 localhost sshd[528649]: Received disconnect from 139.162.133.194 port 42046:11: Bye Bye [preauth]
# Nov 10 00:11:35 localhost sshd[528649]: Disconnected from invalid user changcan 139.162.133.194 port 42046 [preauth]
# Nov 15 00:08:14 localhost kernel: [524765.106559] br-ced8132414aa: port 2(veth9bfc8bc) entered disabled state
# Nov 15 00:08:14 localhost kernel: [524765.108180] br-ced8132414aa: port 2(veth9bfc8bc) entered disabled state
# Nov 15 00:08:14 localhost kernel: [524765.110278] device veth9bfc8bc left promiscuous mode
# Nov 15 00:08:14 localhost kernel: [524765.110284] br-ced8132414aa: port 2(veth9bfc8bc) entered disabled state
# 2024/09/23 15:05:44 [emerg] 639395#0: open() "/www/server/nginx/src/conf/uwsgi_params" failed (2: No such file or directory) in /www/server/panel/vhost/nginx/bgsub_server.conf:14
# www.pipixia.org---100.122.17.191 - - [06/Oct/2024:03:46:21 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white,green,blue,red&sort=_default HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=black%2Cwhite%2Cgreen%2Cblue%2Cred" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.130.32"
# www.pipixia.org---100.122.17.142 - - [06/Oct/2024:03:48:25 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=blue%2Cblack%2Cwhite" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.151.174"

print(ans['text'])
