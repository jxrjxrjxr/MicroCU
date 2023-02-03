import sys
if ".." not in sys.path:
    sys.path.append("..")
from data.fill_data import fillData
from algo.granger_causal import grangerCausal, grangerCausalmp
from algo.build_graph import buildGraph
from algo.analyze_root import analyzeRootDyCause
from dutils.metric import displayRes


def rcaDyCause(cData, _head, config):
    fData = fillData(cData, config)
    grangerRes = grangerCausalmp(fData, _head, config)
    matrix, peak = buildGraph(grangerRes, config)
    nodesDyCause = analyzeRootDyCause(matrix, cData, config)
    metricsDyCause = displayRes(nodesDyCause, config, display=False)
    return metricsDyCause, (nodesDyCause,)
