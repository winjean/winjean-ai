安装依赖  
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple 

pip install --upgrade setuptools

## 清理缓存
pip cache purge

## 列出已安装的包
pip list

## 删除指定的包
pip uninstall [package-name] 

## 查看包信息
pip show [package-name]

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config list

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package-name]

pip install autogenstudio -i https://mirrors.aliyun.com/pypi/simple