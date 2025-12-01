#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: setup.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/04/24
"""
# setup.py
from setuptools import setup, find_packages

setup(
    name='spoton_core',
    version='0.1.0',
    description='Spot-On 核心算法剥离包',
    packages=find_packages(),        # 会自动找到 spoton_core
    install_requires=[
        'numpy>=1.15',
        'scipy>=1.1',
        'lmfit>=1.0',
    ],
    python_requires='>=3.7',
)