#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: __init__.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/04/24
"""
from .fastspt import *
from .version import __version__

from . import fastSPT_tools, readers, writers

try:
	from . import fastSPT_plot
except Exception as e:
	print("Could not import the plot submodule, error:")
	print(e)