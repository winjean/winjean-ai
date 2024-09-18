conda config --show  查看envs_dirs
conda config --show channels

conda config --add envs_dirs D:/.conda

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes


conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda create --name [env-name]: 创建一个新的环境。
conda create -n test pyton==3.9

conda activate [env-name]: 激活指定环境。

conda deactivate: 退出当前环境。
conda env list: 列出所有可用的环境。
conda info --env (同上)
conda list: 在当前环境中列出所有已安装的包。
conda list [package-name]: 在当前环境中列出对应已安装的包。
conda install [package-name]: 安装指定的包。
conda update [package-name]: 更新指定的包。
conda remove [package-name]: 移除指定的包。
conda search [package-name]: 搜索可用的包版本。

  
conda env export > environment.yml: 导出当前环境的配置到一个YAML文件。
conda env create -f environment.yml: 使用YAML文件创建一个新环境。
conda update conda: 更新 Conda 到最新版本。
conda info: 显示关于 Conda 的信息。


