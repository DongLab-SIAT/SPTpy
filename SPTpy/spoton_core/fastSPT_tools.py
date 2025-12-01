#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: fastSPT_tools.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/04/24
"""
## fastSPT_tools
## Some tools for the fastSPT package
## By MW, GPLv3+
## March 2017

## ==== Imports
import pickle, sys, scipy.io, os
from importlib import reload
import numpy as np

####在指定目录path下，如果有一个dataset.py文件，就调用它的list函数，获取可用数据集的列表
## ==== Sample dataset-related functions
def list_sample_datasets(path):
    """Simple relay function that allows to list datasets from a datasets.py file"""
    sys.path.append(path)
    import datasets
    reload(datasets)  # Important I think
    return datasets.list(path, string=True)

###实际读取.mat文件，从变量trackPar中提取轨迹数组；最后把多个细胞的轨迹按列拼接（hstack），返回一个大的Numpy二维数组，形如（N_steps,2）
###(N_particles,...),具体取决于datasets.py的定义
def load_dataset(path, datasetID, cellID):
    """Simple helper function to load one or several cells from a dataset"""
    ## Get the information about the datasets
    sys.path.append(path)
    import datasets
    reload(datasets)  # Important I think
    li = datasets.list(path, string=False)

    if type(cellID) == int:
        cellID = [cellID]

    try:  ## Check if our dataset(s) is/are available
        for cid in cellID:
            if not li[1][datasetID][cid].lower() == "found":
                raise IOError(
                    "This dataset does not seem to be available. Either it couldn't be found or it doesn't exist in the database.")
    except:
        raise IOError(
            "This dataset does not seem to be available. Either it couldn't be found or it doesn't exist in the database or there is a problem with the database.")

    da_info = li[0][datasetID]

    ## Load the datasets
    AllData = []
    for ci in cellID:
        mat = scipy.io.loadmat(os.path.join(path,
                                            da_info['path'],
                                            da_info['workspaces'][ci]))
        AllData.append(np.asarray(mat['trackedPar'][0]))
    return np.hstack(AllData)  ## Concatenate them before returning

####当你已经知道.mat文件的完整路径时，直接调用它即可载入这个单文件样本的轨迹数据。
####同样是trackedPar变量，经np.asarray(...)[0]处理后得到纯粹的NumPy数组。
def load_dataset_from_path(path):
    """Returns a dataset object from a Matlab file"""
    mat = scipy.io.loadmat(path)
    return np.asarray(mat['trackedPar'][0])
