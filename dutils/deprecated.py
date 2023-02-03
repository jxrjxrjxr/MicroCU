def MicroCInorm(_rInfo):
    nodesMicroCI = _rInfo[0]
    nodesMicroCIn = scoreNorm(nodesMicroCI)
    metrics = displayRes(nodesMicroCIn, config, display=False)
    return metrics, (nodesMicroCIn,)