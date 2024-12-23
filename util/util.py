import collections
import numpy as np

# 定义XYZ坐标元组
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def xyz2irc(xyz_tuple, origin_xyz, vxSize_xyz, direction_a):
    """
    将物理坐标(xyz)转换为图像坐标(irc)
    
    Args:
        xyz_tuple: 物理空间中的坐标点
        origin_xyz: 图像原点在物理空间中的坐标
        vxSize_xyz: 体素大小
        direction_a: 方向矩阵
    
    Returns:
        转换后的图像坐标(irc)
    """
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = np.array(xyz_tuple)
    
    # 计算相对于原点的偏移
    coords_offset_a = coords_xyz - origin_a
    
    # 应用方向矩阵的逆变换
    coords_irc = np.matmul(np.linalg.inv(direction_a), coords_offset_a)
    
    # 除以体素大小得到最终的图像坐标
    coords_irc = coords_irc / vxSize_a
    
    # 转换为整数坐标
    coords_irc = np.round(coords_irc)
    
    return tuple(coords_irc.astype(np.int16))