# CT数据处理模块使用文档

## 目录
1. [概述](#1-概述)
2. [数据集模块 (dsets.py)](#2-数据集模块-dsetspy)
3. [数据变换模块 (transforms.py)](#3-数据变换模块-transformspy)
4. [使用示例](#4-使用示例)
5. [注意事项与限制](#5-注意事项与限制)

## 1. 概述

本模块提供了CT图像数据的加载、处理和变换功能。主要包含两个核心组件：
- `dsets.py`: 负责数据集的加载和管理
- `transforms.py`: 负责数据的变换和处理

## 2. 数据集模块 (dsets.py)

### 2.1 核心类和方法

#### LunaDataset 类
```python
dataset = LunaDataset(
    val_stride=10,          # 验证集步长，默认0
    isValSet_bool=None,     # 是否为验证集
    series_uid=None,        # 指定CT序列ID
    transform=None          # 数据变换函数
)
```

**参数说明：**
- `val_stride`: 验证集采样步长，必须 > 0
- `isValSet_bool`: True表示验证集，False表示训练集
- `series_uid`: 指定处理单个CT序列
- `transform`: 数据变换函数

**重要属性：**
- `candidateInfo_list`: 候选结节信息列表
- `transform`: 数据变换函数

#### getCt 函数
```python
ct = getCt(series_uid)  # 获取CT实例
```

**参数说明：**
- `series_uid`: CT序列的唯一标识符

**返回值：**
- 返回Ct类实例，包含CT数据和元信息

### 2.2 数据格式

#### 输入数据要求
- `.mhd`文件：CT图像元数据
- `.raw`文件：CT图像数据
- `annotations.csv`：结节标注信息
- `candidates.csv`：候选结节信息

#### 输出数据格式
```python
(candidate_t, pos_t, series_uid, center_irc) = dataset[index]
```
- `candidate_t`: 形状为(1, 32, 48, 48)的张量
- `pos_t`: 二元标签[not_nodule, is_nodule]
- `series_uid`: CT序列ID
- `center_irc`: 中心坐标

### 2.3 缓存管理
```python
dataset.clear_cache()  # 清理缓存
```

## 3. 数据变换模块 (transforms.py)

### 3.1 TransformBuilder 类

#### 基本用法
```python
builder = TransformBuilder()
transform = builder\
    .add_normalize(min_val=-1000, max_val=1000)\
    .add_window(center=40, width=400)\
    .build()
```

#### 内置变换方法
1. **归一化 (add_normalize)**
```python
add_normalize(min_val=-1000.0, max_val=1000.0)
```
- `min_val`: 最小HU值
- `max_val`: 最大HU值

2. **窗位窗宽处理 (add_window)**
```python
add_window(center=40, width=400)
```
- `center`: 窗位中心值
- `width`: 窗宽范围

3. **自定义变换 (add_custom)**
```python
add_custom(func, name)
```
- `func`: 变换函数
- `name`: 变换名称

### 3.2 快速创建变换
```python
transform = create_transform(
    normalize=True,    # 是否使用归一化
    window=False,      # 是否使用窗位窗宽处理
    custom_func=None   # 自定义处理函数
)
```

## 4. 使用示例

### 4.1 基本数据加载
```python
# 创建数据集
dataset = LunaDataset(val_stride=10, isValSet_bool=False)

# 获取单个样本
sample, label, uid, center = dataset[0]
```

### 4.2 使用数据变换
```python
# 创建变换函数
transform = TransformBuilder()\
    .add_normalize()\
    .add_window(center=40, width=400)\
    .build()

# 在数据集中使用
dataset = LunaDataset(
    transform=transform,
    val_stride=10
)
```

### 4.3 自定义变换
```python
def custom_transform(x):
    return x * 2

transform = TransformBuilder()\
    .add_normalize()\
    .add_custom(custom_transform, "double")\
    .build()
```

## 5. 注意事项与限制

### 5.1 禁止事项
1. **不要在transform函数中修改原始数据**
```python
# 错误示例
def bad_transform(x):
    x *= 2  # 直接修改输入
    return x

# 正确示例
def good_transform(x):
    return x * 2  # 返回新的数据
```

2. **避免在transform中使用外部状态**
```python
# 错误示例
global_var = 1.0
def bad_transform(x):
    return x * global_var

# 正确示例
def create_transform(factor):
    return lambda x: x * factor
```

3. **不要在transform中进行形状改变**
- 保持输入输出维度一致
- 使用custom_collate处理批次大小不一致的情况

### 5.2 性能考虑
1. 合理使用缓存
2. 及时清理不需要的缓存
3. 避免过多的变换步骤

### 5.3 最佳实践
1. 按照处理逻辑顺序添加变换
2. 保持变换函数的简单性
3. 使用合适的数据类型
4. 注意数值范围的控制

## 6. 调试与错误处理

### 6.1 常见问题
1. 内存溢出：及时清理缓存
2. 数据类型不匹配：确保transform返回正确的类型
3. 维度错误：检查数据形状

### 6.2 调试技巧
1. 使用小数据集进行测试
2. 打印中间结果
3. 检查数值范围