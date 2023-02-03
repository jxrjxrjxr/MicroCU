# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python [conda env:tsai]
#     language: python
#     name: tsai
# ---

from dutils.loaddata import mask
import numpy as np


def printMissingRate(data):
    print((data != 0).sum() / np.size(data))


def maskData(data, config):
    sr = config["Data params"]["sample_rate"]
    useNewMask = config["Data params"]["useNewMask"]
    mode = config["Data params"]["maskMode"]
    if useNewMask:
        mData, dataMask = mask(data, sample_rate=sr, mode=mode)
    else:
        dataMask = np.load(config["Data params"]["mask_input"])
        mData = data * dataMask
    return mData, dataMask
