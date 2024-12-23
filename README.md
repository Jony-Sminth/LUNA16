# LUNA16
A repository only for learning LUNA16.
not for any commercial purpose just for learning purpose.

- project_root/
    - README.md            # 项目说明文档
    - requirements.txt        # 项目所需库
    - tests/                # 测试代码
    - util/                # 工具代码
        - disk.py            # 内存缓存系统，缓存函数的返回结果，避免重复计算
        - logconf.py         # 统一的日志配置，帮助记录和调试程序运行过程
        - util.py           # 坐标转换相关
    - data/
        - annotations.csv        # 标注文件
        - candidates.csv         # 候选文件
    - dataProcessing/
        - data_process.py    # 数据处理代码
    - requirements.txt          # 项目所需库
    - .gitignore            # Git 忽略文件
```