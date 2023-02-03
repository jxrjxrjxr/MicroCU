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

from .rand_walk import randWalk
from .rank_node import rankNode
from .ranknode import analyze_root_dycause


def analyzeRootABC(matrix, peak, data, config):
    outPath = randWalk(matrix, peak, config)
    nodes = rankNode(matrix, data, outPath, config)
    for u in range(len(nodes)):
        nodes[u][0] += 1
    return nodes, (outPath)


def analyzeRootDyCause(matrix, data, config):
    nodes = analyze_root_dycause(
        matrix,
        config["rw params"]["frontend"],
        data,
        mean_method=config["rw params"]["mean_method"],
        max_path_length=config["rw params"]["max_path_length"],
        topk_path=config["rw params"]["topk_path"],
        prob_thres=config["rw params"]["prob_thres"],
        num_sel_node=config["rw params"]["num_sel_node"],
        verbose=False
    )
    return nodes