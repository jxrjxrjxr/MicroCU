import time
import os
from shutil import copy
import pandas as pd
import numpy as np
from functools import reduce
from .metric import displayRes


def createExpr(methodList, times, saveRes=True):
    methodStr = reduce(lambda x, y: x + '-' + y, methodList)
    localtime = time.localtime(time.time())
    timeStr = "%02d%02d.%02d%02d.%02d%02d" % (localtime[1], localtime[2], localtime[3], localtime[4], localtime[6], localtime[5])
    exprDirName = methodStr + '-' + str(times) + '-' + timeStr
    resDir = '/workspace/code/MicroCU/results'
    exprDir = os.path.join(resDir, exprDirName)
    if saveRes:
        os.makedirs(exprDir, exist_ok=True)
        copy(os.path.join(resDir, "curExpr.txt"), os.path.join(exprDir, "curExpr.txt"))
    results = pd.DataFrame(np.zeros((len(methodList), 7)))
    results.index = methodList
    results.columns = ['PR@1', 'PR@2', 'PR@3', 'PR@4', 'PR@5', 'PR@avg', 'Acc']
    return exprDir, results


def saveMask(_mask, exprDir, loop):
    np.save(os.path.join(exprDir, "mask{:02d}.npy").format(loop), _mask)


def loadMask(exprDir, loop):
    return np.load(os.path.join(exprDir, "mask{:02d}.npy").format(loop))


def list2csv(methodList):
    length = len(methodList)
    results = pd.DataFrame(np.zeros((length, 7)))
    results.columns = ['PR@1', 'PR@2', 'PR@3', 'PR@4', 'PR@5', 'PR@avg', 'Acc']
    for i in range(length):
        results.iloc[i, :] += methodList[i]
    return results


def adjustRate(nodes, config):
    r = config["rw params"]["rate"]
    newNodes = nodes
    newNodes.sort(key=lambda x: x[1] * r + x[2], reverse=True)
    metrics = displayRes(newNodes, config, display=False)
    return metrics
