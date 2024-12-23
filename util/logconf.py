import logging

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加自定义的日志处理器或过滤器
def getLogger(name):
    logger = logging.getLogger(name)
    # 添加自定义的处理
    return logger

# 导出标准的logging模块
logging.getLogger = getLogger