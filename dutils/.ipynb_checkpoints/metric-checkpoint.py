from dutils.evaluation_function import my_acc, pr_stat, print_prk_acc
import numpy as np


def displayRes(nodes, config, display=True):
    trueRootCause = config["rw params"]["true_root_cause"]
    varNum = config["Data params"]["varNum"]
    prkS_list, acc_list = [], []
    prkS = pr_stat(nodes, trueRootCause)
    acc = my_acc(nodes, trueRootCause, varNum)
    prkS_list.append(prkS)
    acc_list.append(acc)
    prkS = np.mean(np.array(prkS_list), axis=0).tolist()
    acc = float(np.mean(np.array(acc_list)))
    metrics = prkS + [np.mean(prkS), acc]
    if display:
        print_prk_acc(prkS, acc)
    return metrics
