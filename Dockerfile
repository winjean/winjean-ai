# 使用官方的 Python 3.11 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 配置 pip 使用清华大学的镜像源
RUN echo '[global]' > /etc/pip.conf && \
    echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple/' >> /etc/pip.conf && \
    echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> /etc/pip.conf

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到容器中
COPY . /app

# 设置环境变量
ENV FASTAPI_ENV=production

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t my-fastapi-app .
#docker run -d -p 8000:8000 my-fastapi-app
