import warnings
from algo.analyze_root import analyzeRootABC, analyzeRootDyCause
from algo.build_graph import buildGraph
from algo.granger_causal import grangerCausal
from algo.rank_node import printNode, scoreNorm
from data.clip_data import clipData
from data.fill_data import fillData
from data.load_data import loadData
from data.mask_data import maskData, printMissingRate
from dutils.metric import displayRes
from params.load_params import loadConfig
from methods.micro_ci import rcaMicroCI

# %reload_ext autoreload
# %autoreload 2

warnings.filterwarnings("ignore")
import time
st = time.time()
config = loadConfig()
config["Data params"]["sample_rate"] = 0.4
length = 200
config["Data params"]["before_length"] = length
config["Data params"]["after_length"] = length
rData, _head = loadData(config)
mData, _mask = maskData(rData, config)
printMissingRate(mData)
cData, _cMask = clipData(mData, config, _mask)
prepare = time.time()
print(prepare - st, 's')
metricsCI, _ = rcaMicroCI(cData, _head, config, True)
print(time.time() - prepare, 's')


