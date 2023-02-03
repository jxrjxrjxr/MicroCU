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

from dutils.loaddata import zeroFill


def fillData(data, config):
    method = config["Data params"]["zeroFillMethod"]
    verbose = config["Debug params"]["verbose"]
    fData = zeroFill(
        data=data,
        normalize=True,
        zero_fill_method=method,
        verbose=False
    )
    return fData
