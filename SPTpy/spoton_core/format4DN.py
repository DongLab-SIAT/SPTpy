#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: format4DN.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/04/24
"""
# -*-coding:utf-8-*-
# Library to read/write in the 4DN format.
# So far, the 4DN format is documented on: https://docs.google.com/document/d/1SKljQyuTNtKQOxOD5AC9ZBqZZETDXtUz1BImGZ99Z3M/edit#
# By MW, GPLv3+, Jun 2018

from . import readers
import pandas as pd
from itertools import takewhile

###读取所有以#开头的行，检查格式声明行是否合法，提取格式版本号
###检查格式声明行是否合法，提取格式版本号
###解析剩余行，形如key：alue，去掉#并以Key：Value存入字典
###返回dict，至少包含version,以及用户在头部注明的所有元数据字段
def read_header(fn):
    """Function takes a file name and returns the header of the file. The
    function returns a dictionary. Some minimal checks are performed to
    assess whether the SPT file is valid.
    However, the header has to go through the `check_header` function in order
    to be fully validated.
    Part of this function is inspired from: https://stackoverflow.com/a/39724905/9734607)
    """

    def read_raw_header(fn):
        with open(fn, 'r') as fobj:
            # takewhile returns an iterator over all the lines that start with the comment string
            headiter = takewhile(lambda s: s.startswith('#'), fobj)
            # you may want to process the headers differently,  but here we just convert it to a list
            header = [i.replace('\n', '') for i in headiter]
        return header

    hd = read_raw_header(fn)
    out = {}

    # Check consistency
    if len(hd) == 0:
        raise IOError("Invalid file format: the file contains no metadata.")

    # Check SPT format
    if not hd[0].startswith("##SPT format v"):
        raise IOError("Invalid file format: the first line of metadata should start as: '##SPT format v'.")
    else:
        out['version'] = hd[0].replace("##SPT format v", "")
    for i in range(2, len(hd)):
        if ':' not in hd[i]:
            raise IOError("Missing colon ':' in line {} ({}).".format(i + 1, hd[i]))
        elif ' ' in hd[i].split(":"):
            raise IOError("Space exist in the name of the header on line {} ({})".format(i + 1, hd[i]))
        else:
            s = hd[i].split(":")
            out[s[0][1:]] = (":".join(s[1:])).strip()
    return out

####当需要新建一个符合4DN规范的文件头时，可先用此函数生成一个字典，然后用户可逐项填入实际值
def get_empty_header(version='1.0', desc=True, minimal=True):
    """Function returns a valid empty header that could then be edited.
    Arguments:
        version: (string) should match an available version of the format
        desc: True if the description of the fields should be returned
        minimal: True if only the mandatory fields should be returned
    """
    # Check version
    versions = ['1.0']
    if version not in versions:
        raise TypeError("Unsupported SPT format version, the supported versions are: {}".format(", ".join(versions)))

    # The meat
    if version == '1.0':
        di = {
            "version": "## SPT format v1.0",
            "#TimeUnit": "s (e.g. 's' for second, 'ms' for millisecond, 'min' for minutes and 'hr' for hours).",
            "#XYZUnit": "e.g. um (use 'um' for μm to avoid problem with special, Greek symbols, could also be 'nm' for nanometers, 'm' for meters or whatever is appropriate). X,Y,Z, and error units are considered to be the same.",
            "#TimeIncrement": "time between frames in same time units as above; necessary to distinguish trajectories with gaps",
            "#columns:": "Trajectory_ID\X\Y\ etc",
            "#LocalizationErrorDefinition": "Required if localization errors are provided. acceptable terms: 'std', 'FWHM', 'median'",
            "#LocalizationError_Algorithm_ID": "Required if localization errors are provided. 'First Author last name', 'year', DOI (e.g. Quan, 2010, doi:10.1117/1.3505017)",
        }
        di2 = {  # Optional arguments
            "#Image_SizeT": "integer (the number of time frames used for tracking)",
            "#Image_SizeC": "integer (the number of channels used for tracking). Typically the number of colors. If only 1 color, you can leave this out.",
            "#Channel_ID": "integer, matching the integer used in the data columns and referring to the metadata Channel object.",
            "#Channel_Name": "string, e.g. Tubulin or FLAG-Halo-CTCF",
            "#Channel_Fluorofore": "string, e.g. GFP or Halo-TMR",
            "#Channel_Color": "string, e.g. Green",
            "#Channel_ExcitationWavelength": "integer, e.g. 488 (default units: nm)",
            "#Channel_EmissionWawelength": "integer, e.g. 525 (default units: nm) ■ Note: insert one set of channel specifications for each channel used for tracking. ■ Note: if more than one Channel is utilized please use a progressive number to indicate the different sets of channel specifications. ■ Not that if only one color channel is used, it is not necessary to fill this out.",
            "#ROI_ID": "unique identifier",
            "#ROI_Name": "string, e.g. nucleolus, nuclear speckles, mitochondria.",
            "#ROI_Description": "string; ideally a description of how the ROIs were segmented (e.g. HP1-GFP) and maybe a description of what the name is of the file containing the segmentation mask (if appropriate) and criteria for judging whether or not a localization was inside or outside the compartment. ■ Note: insert one set of ROI specifications for each ROI utilized. ■ Note: if more than one ROI is utilized please use a progressive number to indicate the different sets of ROI specifications.",
            "#PixelSize": "(e.g μm/pixel), use same units as for XYZ. E.g.: #PixelSize: 0.16",
            "#MaxGaps": "integer: 0,1,2...): maximal number of gaps allowed in the tracking",
            "#ParticleLocalization_Algorithm": "?",
            "#ParticleLinking_Algorithm": "these two describe the algorithm(s) used for localization and/or linking and is an optional field. In case the same algorithm or program does both localization/detection 6and linking/tracking, you can add the same information to both. They have the following optional attributes:",
            "#ParticleLocalization_Algorithm_ID": "unique identifier",
            "#Algorithm_Name": "as provided by authors; e.g. MTT algorithm",
            "#Algorithm_AuthorName": "e.g. EK Smith and JK Johnson",
            "#Algorithm_Version": "e.g. 1.07",
            "#Algorithm_Publication": "e.g. I. F. Sbalzarini and P. Koumoutsakos. Feature Point Tracking and Trajectory Analysis for Video Imaging in Cell Biology, Journal of Structural Biology 151(2):182-195, 2005.",
            "#Algorithm_ReleaseDate": "e.g. 2005-07-22",
            "#Algorithm_Description": "could be a long string/paragraph. Could also describe specific parameters used in the algorithm. As an example of how to string together e.g. #ParticleLinking_Algorithm and its attributes, if we want to list its publication we would use '#ParticleLinking_Algorithm_Publication'.",
            "#PhotonCount_CalibrationMethod": "acceptable terms: 'camera' or 'manual'",
            "#PhotonCount_Gain": "numeric, double"
        }
    else:
        raise TypeError("Uncatched error.")

    if not minimal:  # merge the two dictionary to get the "maximal" specification.
        for (k, kk) in di2.items():
            di[k] = kk

    # Remove descriptions if needed
    if not desc:
        return {k: "" for k in di.keys()}
    else:
        return di

###对比用户传入的hd字典与get_empty_header(minimal=True)返回的“必需字段”。
###还会检查hd['columns']中的列名是否覆盖实际数据列
###返回[True]或[False,错误1,错误2,...]
def check_header(hd, columns, check_columns=True):
    """This function checks that we are working with a valid header.
    Arguments:
        hd: a header dictionary (as returned by `get_empty_header` or `read_header`)
    """
    versions = ['1.0']
    ok = []

    if hd['version'] not in versions:  # Check the SPT format version
        ok.append("The version of the SPT format is not supported.")

    # Check that the mandatory fields are present
    req = get_empty_header(hd['version'], desc=False, minimal=True)
    print(req.keys())
    print(hd.keys())
    for r in req.keys():
        if r[1:] not in hd:
            if r != "version":
                ok.append("Mandatory key {} is not present in the header".format(r))

    # List the additional fields
    new_fields = []
    for r in hd.keys():
        if r not in req:
            new_fields.append(r)

    # Check that the columns are properly described
    if check_columns:
        col = hd["columns"].split("\\")
        for c in columns:
            if c not in col:
                ok.append("Column '{}' exists but is not described in the 'columns' field.".format(c))

    if len(ok) == 0:
        return [True]
    else:
        return [False] + ok

####读取一个4DN格式文件，返回fastSPT-compatible数据结构，或可选地返回pandas.DataFrame和header字典。
def read_4DN(fn, return_header=False, return_pandas=False):
    """Main function to read a 4DN format. This function returns the data in a
    format compatible with fastSPT. Alternatively, it can provide the data in a
    panda dataframe.
    """

    def parse_columns(col):
        return col.split("\t")

    hd = read_header(fn)
    cols = parse_columns(hd["Columns"])
    df = pd.read_csv(fn, sep='\t', comment='#', names=cols)
    if len(df["Cell_ID"].unique()) != 1 or len(df["BiologicalReplicate_ID"].unique()) != 1:
        raise IOError(
            "4DN format with more than one Cell_ID or BiologicalReplicate_ID is currently not supported. You might want to split your file using an online tool such as: https://tjian-darzacq-lab.gitlab.io/Split_4DN_format/")

    if not return_pandas:  # convert to fastSPT format
        try:  # try to read properly formatted file
            df = readers.pandas_to_fastSPT(df, 'Trajectory_ID', 'X', 'Y', 't', 'Frame_ID')
        except:  # Read buggy format
            df = readers.pandas_to_fastSPT(df, 'Trajectory_ID', ' X', ' Y', ' t', ' Frame_ID')  # contains spaces
    if return_header:
        return {"header": hd, "data": df}
    else:
        return df
