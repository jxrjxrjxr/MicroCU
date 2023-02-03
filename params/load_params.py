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

import os
import json
import time


def loadConfig():
    with open(os.path.join("params", "config.json"), "r") as loadConfigF:
        config = json.load(loadConfigF)
    localtime    = time.localtime(time.time())
    timeStr      = "%02d%02d.%02d%02d.%02d%02d"%(localtime[1], localtime[2], localtime[3],
                                                 localtime[4], localtime[6], localtime[5])
    outFile = "{}-{}-{}-{}-{}-{}".format(str(config["Data params"]["before_length"]),
                                         str(config["Data params"]["after_length"]),
                                         str(config["Data params"]["sample_rate"]),
                                         str(config["Granger params"]["auto_threshold_ratio"]),
                                         str(config["Granger params"]["step"]),
                                         timeStr)
    dirOutput = os.path.join("results", outFile)
    config["Debug params"]["dirOutput"] = dirOutput
    return config


def dumpConfig(config, expDir):
    with open(os.path.join(expDir, "config.json"), "w") as dumpConfigF:
        json.dump(config, dumpConfigF)