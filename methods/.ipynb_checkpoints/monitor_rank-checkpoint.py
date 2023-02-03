import os
import random
import time
import sys

if ".." not in sys.path:
    sys.path.append("..")

import numpy as np
from algo.get_matrix_by_pc import buildGraphPC
from dutils.pearson import calc_pearson
from dutils.loaddata import aggregate
from dutils.metric import displayRes
from data.fill_data import fillData


def firstorder_randomwalk(
    P,
    epochs,
    start_node,
    teleportation_prob,
    label=[],
    walk_step=1000,
    print_trace=False,
):
    n = P.shape[0]
    score = np.zeros([n])
    current = start_node - 1
    for epoch in range(epochs):
        if print_trace:
            print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(P[current]) == 0:
                current = np.random.choice(range(n), p=teleportation_prob)
                break
            else:
                next_node = np.random.choice(range(n), p=P[current])
                if print_trace:
                    print("{:2d}".format(current + 1), end="->")
                score[next_node] += 1
                current = next_node
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


def normalize(p):
    """Normalize the matrix in each row
    """
    p = p.copy()
    for i in range(p.shape[0]):
        row_sum = np.sum(p[i])
        if row_sum > 0:
            p[i] /= row_sum
    return p


def relaToRank(rela, access, rankPaces, frontend, rho=0.3, print_trace=False):
    n = len(access)
    S = [abs(_) for _ in rela[frontend - 1]]
    P = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            # forward edge
            if access[i][j] != 0:
                P[i, j] = abs(S[j])
            # backward edge
            elif access[j][i] != 0:
                P[i, j] = rho * abs(S[i])
    # Add self edges
    for i in range(n):
        if i != frontend - 1:
            P[i][i] = max(0, S[i] - max(P[i]))
    P = normalize(P)

    teleportation_prob = (np.array(S) / np.sum(S)).tolist()
    label = [i for i in range(1, n + 1)]
    l = firstorder_randomwalk(
        P, rankPaces, frontend, teleportation_prob, label, 
        print_trace=print_trace
    )
    # print(l)
    return l, P


def rcaMonitorRank(cData, _head, config, display=False):
    pc_aggregate = config["PC params"]["pc_aggregate"]
    pc_alpha = config["PC params"]["pc_alpha"]
    frontend = config["rw params"]["frontend"]
    fData = fillData(cData, config)
    data = np.array([aggregate(row, pc_aggregate) for row in fData.T])
    rela = calc_pearson(data, method="numpy", zero_diag=False)
#     import pdb; pdb.set_trace()
    dep_graph = buildGraphPC(data, alpha=pc_alpha)
#     from algo.draw_graph import draw_graph_gt
#     draw_graph_gt(dep_graph, _head)
    nodesMR, P = relaToRank(rela, dep_graph, 10, frontend, rho=0.2, print_trace=False)
    metricsMR = displayRes(nodesMR, config, display=display)
    return metricsMR, (nodesMR,)
