import collections
import datetime
import functools
import hashlib
import pickle
import random
import string

# 缓存类实现
class Cache:
    def __init__(self, name):
        self.name = name
        self._dict = {}
        self._hash_dict = collections.defaultdict(list)

    def memoize(self, typed=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 创建键值
                key = (args, tuple(sorted(kwargs.items())))
                if typed:
                    key += tuple(type(arg) for arg in args)
                    key += tuple(type(value) for _, value in sorted(kwargs.items()))
                
                # 尝试从缓存获取
                try:
                    return self._dict[key]
                except KeyError:
                    # 如果没有缓存，执行函数并存储结果
                    value = func(*args, **kwargs)
                    self._dict[key] = value
                    return value
            return wrapper
        return decorator

def getCache(name):
    return Cache(name)