安装依赖  
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple 

pip install --upgrade setuptools

pip cache purge

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package-name]

pip install autogenstudio -i https://mirrors.aliyun.com/pypi/simple