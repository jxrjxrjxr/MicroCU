# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python3.6-0522
#     language: python
#     name: python3.6-0522
# ---

from collections import defaultdict
import numpy as np
from .causal_graph_build import normalize_by_column, get_overlay_count, get_segment_split


def getPeak(overCount, listSegment, x_i, y_i):
    length = len(listSegment)
    se, ov = listSegment, overCount
    peakList = []
    peakW = []
    gap = se[1] - se[0]
    segNum = 0
    while segNum + 1 < length - 1:
        if (segNum == 0 or ov[se[segNum]] > ov[se[segNum - 1]]) and ov[se[segNum]] >= ov[se[segNum + 1]]:
            startNum = segNum
            while segNum + 1 < length - 1 and ov[se[segNum]] == ov[se[segNum + 1]]:
                segNum += 1
            peakList.append((se[segNum] + se[startNum] + gap) // 2)
            peakW.append(segNum - startNum + 1)
        segNum += 1
    if segNum + 1 == length - 1 and ov[se[segNum]] > ov[se[segNum - 1]]:
        peakList.append(se[segNum] + (gap // 2))
        peakW.append(1)
    assert(-1)
    return sum(mul(peakList, peakW)) // (len(peakList) * sum(peakW)) if len(peakList) > 0 else -1


def mul(list1, list2):
    return list(map(lambda item : item[0] * item[1], zip(list1, list2)))


def buildGraph(grangerRes, config):
    varNum = config["Data params"]["varNum"]
    bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
    local_length = bef + aft
    histSum = defaultdict(int)
    peak = defaultdict(int)
    edge, edgeWeight = [], dict()
    matrix = np.zeros([varNum, varNum])
    list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])
    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            histSum[key] = sum(overlay_counts)
            peak[key] = getPeak(overlay_counts, list_segment_split, x_i, y_i)
    for x_i in range(varNum):
        bar_data = []
        for y_i in range(varNum):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histSum[key])
        bar_data_thres = np.max(bar_data) * config["Granger params"]["auto_threshold_ratio"]
        for y_i in range(varNum):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edgeWeight[(x_i, y_i)] = bar_data[y_i]
    for key, val in edgeWeight.items():
        x, y = key
        matrix[x, y] = val
    matrix = normalize_by_column(matrix)
    return matrix, peak

def buildGraphMul(grangerRes, config):
    varNum = config["Data params"]["varNum"]
    bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
    local_length = bef + aft
    histSum = defaultdict(int)
    peak = defaultdict(int)
    edge, edgeWeight = [], dict()
    matrix = np.zeros([varNum, varNum])
    list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])
    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            histSum[key] = sum(overlay_counts)
            peak[key] = getPeak(overlay_counts, list_segment_split, x_i, y_i)
    for x_i in range(varNum):
        bar_data = []
        for y_i in range(varNum):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histSum[key])
        bar_data_thres = np.max(bar_data) * config["Granger params"]["auto_threshold_ratio"]
        for y_i in range(varNum):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edgeWeight[(x_i, y_i)] = bar_data[y_i]
    for key, val in edgeWeight.items():
        x, y = key
        matrix[x, y] = val
    matrix = normalize_by_column(matrix)
    return matrix, peak
