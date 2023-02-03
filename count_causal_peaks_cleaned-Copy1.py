def get_segment_split(n_sample, step):
    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)
    return list_segment_split


get_segment_split(600, 70)

import warnings
from algo.analyze_root import analyzeRootABC, analyzeRootDyCause
from algo.build_graph import buildGraph, genGraph
from algo.granger_causal import grangerCausal, grangerCausalmp
from algo.rank_node import printNode, scoreNorm
from data.clip_data import clipData
from data.fill_data import fillData
from data.load_data import loadData
from data.mask_data import maskData, printMissingRate
from dutils.metric import displayRes
from params.load_params import loadConfig
from methods.micro_ci import rcaMicroCI
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from algo.causal_graph_build import normalize_by_column, get_overlay_count, get_segment_split
from algo.build_graph import getPeak

# %reload_ext autoreload
# %autoreload 2

config = loadConfig()

lengthList = [100, 150, 200, 250, 300, 350, 400, 450, 500]
grangerResList = [None for item in lengthList]

resPathList = [None for item in lengthList]
resPathListUni = [None for item in lengthList]

for i, befaftlength in enumerate(lengthList):
    config['Data params']['before_length'] = befaftlength
    config['Data params']['after_length'] = befaftlength
    rData, _head = loadData(config)
    mData, _mask = maskData(rData, config)
    printMissingRate(mData)
    cData, _cMask = clipData(mData, config, _mask)
    fData = fillData(cData, config)
    grangerRes = grangerCausalmp(fData, _head, config)
    grangerResList[i] = grangerRes
    varNum = config["Data params"]["varNum"]
    bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
    local_length = bef + aft
    orimtx = np.zeros([local_length, varNum, varNum])
    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            orimtx[:, x_i, y_i] = overlay_counts[:, 0]
    list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])
    print(list_segment_split)
    peakPosList = [[[None] for y_i in range(varNum)] for x_i in range(varNum)]
    peakWeiList = [[[None] for y_i in range(varNum)] for x_i in range(varNum)]

    for x_i in range(varNum):
        for y_i in range(varNum):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = grangerRes[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)

            overCount = overlay_counts
            listSegment = list_segment_split
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
                    if segNum + 1 < length - 1 and ov[se[segNum]] < ov[se[segNum + 1]]: continue
                    peakList.append((se[segNum] + se[startNum] + gap) // 2)
                    peakW.append(segNum - startNum + 1)
                segNum += 1
            if segNum + 1 == length - 1 and ov[se[segNum]] > ov[se[segNum - 1]]:
                peakList.append(se[segNum] + (gap // 2))
                peakW.append(1)
            peakPosList[x_i][y_i] = np.array(peakList)
            peakWeiList[x_i][y_i] = np.array(peakW)
    matrix, peak = buildGraph(grangerRes, config)
    resPaths = []

    import queue
    q = queue.Queue()
    q.put([[config["rw params"]["frontend"], local_length]])
    while not q.empty():
        path = q.get()
        lasts = q.qsize()
        las = path[-1][0]
        prevPos = path[-1][1]
        for x_i in range(varNum):
            if matrix[x_i][las] == 0: continue
            if peakWeiList[x_i][las][0] == length - 1: continue
            flag = False
            for pathnode in path:
                if x_i == pathnode[0]:
                    flag = True
                    break
            if flag: continue
            peakPos = peakPosList[x_i][las]
            for pos in peakPos:
                if pos <= prevPos:
                    q.put(path + [[x_i, pos]])
        if q.qsize() == lasts:
            resPaths.append(path)

    resPathList[i] = resPaths

    resPathsUnimodal = []

    import queue
    q = queue.Queue()
    q.put([[config["rw params"]["frontend"], local_length]])
    while not q.empty():
        path = q.get()
        lasts = q.qsize()
        las = path[-1][0]
        prevPos = path[-1][1]
        for x_i in range(varNum):
            if matrix[x_i][las] == 0: continue
            flag = False
            for pathnode in path:
                if x_i == pathnode[0]:
                    flag = True
                    break
            if flag: continue
            pos = peak[f"{x_i}->{las}"]
            if pos <= prevPos:
                q.put(path + [[x_i, pos]])
        if q.qsize() == lasts:
            resPathsUnimodal.append(path)

    resPathListUni[i] = resPathsUnimodal

for i in range(len(lengthList)):
    print(f"{lengthList[i]:3d} {resPathList[i]:5d} {resPathListUni[i]:5d}")

length = 300
config['Data params']['before_length'] = length

rData, _head = loadData(config)
mData, _mask = maskData(rData, config)
printMissingRate(mData)
cData, _cMask = clipData(mData, config, _mask)
fData = fillData(cData, config)

grangerRes = grangerCausalmp(fData, _head, config)

varNum = config["Data params"]["varNum"]
bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
local_length = bef + aft
orimtx = np.zeros([local_length, varNum, varNum])
for x_i in range(varNum):
    for y_i in range(varNum):
        if x_i == y_i:
            continue
        key = "{0}->{1}".format(x_i, y_i)
        intervals = grangerRes[key]["intervals"]
        overlay_counts = get_overlay_count(local_length, intervals)
        orimtx[:, x_i, y_i] = overlay_counts[:, 0]

# +
list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])

peakPosList = [[[None] for y_i in range(varNum)] for x_i in range(varNum)]
peakWeiList = [[[None] for y_i in range(varNum)] for x_i in range(varNum)]

for x_i in range(varNum):
    for y_i in range(varNum):
        if x_i == y_i:
            continue
        key = "{0}->{1}".format(x_i, y_i)
        intervals = grangerRes[key]["intervals"]
        overlay_counts = get_overlay_count(local_length, intervals)
        
        overCount = overlay_counts
        listSegment = list_segment_split
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
                if segNum + 1 < length - 1 and ov[se[segNum]] < ov[se[segNum + 1]]: continue
                peakList.append((se[segNum] + se[startNum] + gap) // 2)
                peakW.append(segNum - startNum + 1)
            segNum += 1
        if segNum + 1 == length - 1 and ov[se[segNum]] > ov[se[segNum - 1]]:
            peakList.append(se[segNum] + (gap // 2))
            peakW.append(1)
        peakPosList[x_i][y_i] = np.array(peakList)
        peakWeiList[x_i][y_i] = np.array(peakW)
# -

matrix, peak = buildGraph(grangerRes, config)

# +
resPaths = []

import queue
q = queue.Queue()
q.put([[config["rw params"]["frontend"], local_length]])
while not q.empty():
    path = q.get()
    lasts = q.qsize()
    las = path[-1][0]
    prevPos = path[-1][1]
    for x_i in range(varNum):
        if matrix[x_i][las] == 0: continue
        if peakWeiList[x_i][las][0] == length - 1: continue
        flag = False
        for pathnode in path:
            if x_i == pathnode[0]:
                flag = True
                break
        if flag: continue
        peakPos = peakPosList[x_i][las]
        for pos in peakPos:
            if pos <= prevPos:
                q.put(path + [[x_i, pos]])
    if q.qsize() == lasts:
        resPaths.append(path)

print(len(resPaths))

# +
resPathsUnimodal = []

import queue
q = queue.Queue()
q.put([[config["rw params"]["frontend"], local_length]])
while not q.empty():
    path = q.get()
    lasts = q.qsize()
    las = path[-1][0]
    prevPos = path[-1][1]
    for x_i in range(varNum):
        if matrix[x_i][las] == 0: continue
        flag = False
        for pathnode in path:
            if x_i == pathnode[0]:
                flag = True
                break
        if flag: continue
        pos = peak[f"{x_i}->{las}"]
        if pos <= prevPos:
            q.put(path + [[x_i, pos]])
    if q.qsize() == lasts:
        resPathsUnimodal.append(path)

print(len(resPathsUnimodal))
