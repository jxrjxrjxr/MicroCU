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

import numpy as np
import matplotlib.pyplot as plt
from algo.causal_graph_build import normalize_by_column, get_overlay_count, get_segment_split
from algo.build_graph import getPeak

# %reload_ext autoreload
# %autoreload 2

warnings.filterwarnings("ignore")
import time
st = time.time()
config = loadConfig()
rData, _head = loadData(config)
mData, _mask = maskData(rData, config)
printMissingRate(mData)
cData, _cMask = clipData(mData, config, _mask)
prepare = time.time()
print(prepare - st, 's')

fData = fillData(cData, config)
grangerRes = grangerCausalmp(fData, _head, config)

grangerRes

aaa = [[1], [2], [0]]
aaa[0] > aaa[1]

varNum = config["Data params"]["varNum"]
bef, aft = config["Data params"]["before_length"], config["Data params"]["after_length"]
local_length = bef + aft
matrix = np.zeros([varNum, varNum])
list_segment_split = get_segment_split(bef + aft, config["Granger params"]["step"])
for x_i in range(varNum):
    for y_i in range(varNum):
        if x_i == y_i:
            continue
        if x_i * varNum + y_i not in res:
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
                peakList.append((se[segNum] + se[startNum] + gap) // 2)
                peakW.append(segNum - startNum + 1)
            segNum += 1
        if segNum + 1 == length - 1 and ov[se[segNum]] > ov[se[segNum - 1]]:
            peakList.append(se[segNum] + (gap // 2))
            peakW.append(1)
        if(len(peakList) > 1):
            print(key, peakList, peakW)


def singlePeakJudge(overCount, listSegment):
    overCount = np.append(overCount, [overCount[-1]])
    length = len(listSegment)
    se, ov = listSegment, overCount
    maxSeg = 0
    for i in range(length):
        if ov[se[i]] > ov[se[maxSeg]]:
            maxSeg = i
    for i in range(0, maxSeg):
        if ov[se[i]] > ov[se[i + 1]]:
            return False
    for i in range(maxSeg, length - 1):
        if ov[se[i]] < ov[se[i + 1]]:
            return False
    return True


cnt

# svarNum = 5
sel = [1, 5, 8, 12, 13, 15, 23, 27, 29, 30, 32]
# fig, ax = plt.subplots(8, 1, figsize=(16, 3*8))
cnt = 0
res = []
for x_i in range(varNum):
    for y_i in range(varNum):
        if x_i == y_i:
            continue
        if not (x_i in sel and y_i in sel):
            continue
        key = "{0}->{1}".format(x_i, y_i)
        intervals = grangerRes[key]["intervals"]
        overlay_counts = get_overlay_count(local_length, intervals)
        if not singlePeakJudge(overlay_counts, list_segment_split):
            res.append(x_i * varNum + y_i)
#             print(key)
#             ax[cnt].plot(overlay_counts)
#             ax[cnt].legend([key])
            cnt += 1
# plt.savefig('../byte-project/results/0801/metrics-multi-peaks-sel.png')

res

plt.plot(mtx[:, 29, 32])

matrix, peak = buildGraph(grangerRes, config)
mtx = genGraph(matrix, peak, config)

mtx.shape

# +
# metricsCI, _ = rcaMicroCI(cData, _head, config, True)
