import sys
if ".." not in sys.path:
    sys.path.append("..")
from data.fill_data import fillData
from algo.granger_causal import grangerCausal, grangerCausalmp
from algo.build_graph import buildGraph
from algo.analyze_root import analyzeRootABC
from dutils.metric import displayRes
from algo.build_graph import genGraph
# from algo.draw_graph import draw_graph_gt

def rcaMicroCI(cData, _head, config, display=False):
    fData = fillData(cData, config)
#     grangerRes = grangerCausal(fData, _head, config)
    grangerRes = grangerCausalmp(fData, _head, config)
    matrix, peak = buildGraph(grangerRes, config)
    mtx = genGraph(matrix, peak, config)
#     draw_graph_gt(mtx, _head, vers='v2')
    nodesABC, _aInfo = analyzeRootABC(matrix, peak, cData, config)
    metricsABC = displayRes(nodesABC, config, display=display)
    return metricsABC, (nodesABC,)
