from os.path import join
from dutils.loaddata import loadWithoutZF


def loadData(config):
    dconf = config["Data params"]
    data, data_head = loadWithoutZF(
        file_path=join("/workspace", "code", "rootCauseAnalyzeTools", "data", dconf["data_source"], "rawdata.xlsx"),
        aggre_delta=dconf["aggre_delta"],
        verbose=config["Debug params"]["verbose"]
    )
    return data, data_head
