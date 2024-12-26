import logging

# 基础配置
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 直接导出 logging.getLogger，不做修改
getLogger = logging.getLogger