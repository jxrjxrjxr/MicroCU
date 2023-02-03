from params.load_params import loadConfig, dumpConfig
from data.load_data import loadData
from data.mask_data import maskData, printMissingRate
from data.fill_data import fillData
from data.clip_data import clipData
from algo.granger_causal import grangerCausal
from algo.build_graph import buildGraph
from algo.analyze_root import analyzeRootABC, analyzeRootDyCause
from dutils.metric import displayRes
from algo.rank_node import scoreNorm, printNode
from methods.micro_ci import rcaMicroCI
from methods.dy_cause import rcaDyCause
from methods.monitor_rank import rcaMonitorRank
from methods.cloud_ranger import rcaCloudRanger
from dutils.expr_utils import createExpr, list2csv, saveMask, adjustRate, loadMask
from dutils.loaddata import zeroFill
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

import warnings
warnings.filterwarnings('ignore')

times = 20
saveRes = True
deprecate = [0, 8, 16]
methodName = ['MonitorRank',
              'CloudRanger',
              'DyCause+linear',
              'DyCause+prla',
              'DyCause+BPMF',
              'MicroCI4:1',
              'MicroCI3:1',
              'MicroCI2:1',
              'MicroCI1:1',
              'MicroCI1:2',
              'MicroCI1:3']
config = loadConfig()
rData, _head = loadData(config)

exprDir, results = createExpr(methodName, times, saveRes=saveRes)
tempDir = os.path.join(".", "results", "tempExpr")
maskDir = os.path.join(".", "data", "maskData-{}".format(config["Data params"]["sample_rate"]))

if saveRes:
    dumpConfig(config, exprDir)
else:
    dumpConfig(config, tempDir)
# rData, _head = loadData(config)
algoTime = {}

for loop in tqdm(range(times)):
    if loop in deprecate:
        continue
    methodNum = 0
#     mData, _mask = maskData(rData, config)
#     if saveRes:
#         saveMask(_mask, exprDir, loop)
    mData = loadMask(maskDir, loop)
    cData, _cMask = clipData(mData, config)
    curTime = time.time()
    # MonitorRank
    mMR, _mrInfo = rcaMonitorRank(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # CloudRanger
    mCR, _crInfo = rcaMonitorRank(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # DyCause+linear
    config["Data params"]["zeroFillMethod"] = "linear"
    mlDyCause, _ldInfo = rcaDyCause(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # DyCause+prla
    config["Data params"]["zeroFillMethod"] = "prevlatter"
    mDyCause, _dInfo = rcaDyCause(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # DyCause+BPMF
    config["Data params"]["zeroFillMethod"] = "BPMF"
    mpDyCause, _pdInfo = rcaDyCause(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # MicroCI4:1
    config["Data params"]["zeroFillMethod"] = "prevlatter"
    config["rw params"]["rate"] = 4
    m41MicroCI, _mInfo = rcaMicroCI(cData, _head, config)
    if not loop:
        algoTime[methodName[methodNum]] = time.time() - curTime
        curTime = time.time()
        methodNum += 1
    # MicroCI3:1
    config["rw params"]["rate"] = 3
    m31MicroCI = adjustRate(_mInfo[0], config)
    # MicroCI2:1
    config["rw params"]["rate"] = 2
    m21MicroCI = adjustRate(_mInfo[0], config)
    # MicroCI1:1
    config["rw params"]["rate"] = 1
    m11MicroCI = adjustRate(_mInfo[0], config)
    # MicroCI1:2
    config["rw params"]["rate"] = 0.5
    m12MicroCI = adjustRate(_mInfo[0], config)
    # MicroCI1:3
    config["rw params"]["rate"] = 0.3333
    m13MicroCI = adjustRate(_mInfo[0], config)
    
    curRes = list2csv([mMR,
                       mCR,
                       mlDyCause,
                       mDyCause,
                       mpDyCause,
                       m41MicroCI,
                       m31MicroCI,
                       m21MicroCI,
                       m11MicroCI,
                       m12MicroCI,
                       m13MicroCI])
    if saveRes:
        curRes.to_csv(os.path.join(exprDir, 'res{:02d}.csv'.format(loop)))
    else:
        curRes.to_csv(os.path.join(tempDir, 'res{:02d}.csv'.format(loop)))
    results = pd.DataFrame(curRes.values + results.values, columns=results.columns, index=results.index)
    phase_res = results / (loop + 1)
    if saveRes:
        phase_res.to_csv(os.path.join(exprDir, 'results.csv'))
    else:
        phase_res.to_csv(os.path.join(tempDir, 'results.csv'))
    if not loop:
        for key in algoTime:
            print("{} : {:.3f}s".format(key, algoTime[key]))