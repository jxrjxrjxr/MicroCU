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

def clipData(data, config, mask=None):
    startT = config["Data params"]["start_time"]
#     import pdb; pdb.set_trace()
    bef = config["Data params"]["before_length"]
    aft = config["Data params"]["after_length"]
    cData = data[startT - bef : startT + aft, :]
    if mask is not None:
        cMask = mask[startT - bef : startT + aft, :]
    else:
        cMask = None
    return cData, cMask
