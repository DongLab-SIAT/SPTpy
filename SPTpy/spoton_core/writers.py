#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: writers.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/04/24
"""
#-*-coding:utf-8-*-
## writers.py is part of the fastspt library
## By MW, GPLv3+, Dec. 2017
## writers.py exports file formats widespread in SPT analysis


## ==== Imports
#import xml.etree.cElementTree as ElementTree
import scipy.io, pandas

###代码注释掉了import xml.etree.cElementTree as ElementTree,而且写死了输出路径，标记为实验性/已废弃，实际上无法直接使用
## ==== Functions
def write_trackmate(da):
    """Experimental (not to say deprecated)"""
    #tree = ElementTree.Element('tmx', {'version': '1.4a'})
    Tracks = ElementTree.Element('Tracks', {'lol': 'oui'})
    ElementTree.SubElement(Tracks,'header',{'adminlang': 'EN',})
    ElementTree.SubElement(Tracks,'body')

    with open('/home/**/Bureau/myfile.xml', 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" ?>')
        ElementTree.ElementTree(Tracks).write(f, 'utf-8')

###将matlab格式的trackedPar(典型的Anders提供的.mat文件)转换为一个带头的CSV
def mat_to_csv(inF, outF):
    """Convert a .mat file (Anders's mat file) to a csv"""
    try:
        d = scipy.io.loadmat(inF)['trackedPar'][0]
    except:
        print("ERROR: "+inF)
        return

    t = []
    for (i, traj) in enumerate(d):
        for j in range(traj[0].shape[0]):
            t.append({'trajectory': i,
                      'x': traj[0][j, 0],
                      'y': traj[0][j, 1],
                      'frame': traj[1][j, 0]-1,
                      't': traj[2][j, 0]})
    pandas.DataFrame(t).to_csv(outF)

##占位符，原计划实现把DataFrame df和头部字典hd导出为4DN格式的文本文件，目前尚未实现。
def write_4dn(df, hd):
    """"""
    raise NotImplementedError("Sorry :s")

##把内存中以[（x,y,t,frame）...]形式存储的多条轨迹，拼成一段完整的csv文本（字符串）。
##返回一个包含首行列名和各轨迹点的多行字符串
def traces_to_csv(traces):
    """Returns a CSV file with the format
    trajectory,x,y,t,frame
    """
    csv = "trajectory,x,y,t,frame\n"
    for (tr_n, tr) in enumerate(traces):
        for pt in tr:
            csv +="{},{},{},{},{}\n".format(tr_n, pt[0],pt[1],pt[2],pt[3])
    return csv

##将traces_to_csv的输出写到磁盘路径fn，生成符合trajectory,x,y,frame格式的csv文件。
def write_csv(fn, da):
    with open(fn, 'w') as f:
        f.write(traces_to_csv(da))

###这里使这个脚本支持命令行调用
###当直接运行.py文件时，取第一个命令行参数（一个.mat路径），自动调用mat_to_csv,并在同目录生成同名的.csv文件。
if __name__ == "__main__":
    import sys
    print("Running standalone")
    mat_to_csv(sys.argv[1], sys.argv[1].replace(".mat", ".csv"))
