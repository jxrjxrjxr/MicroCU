---
# MicroCU
### Descriptions
This is the code repository for "Look Deep into the Microservice System Anomaly through Very Sparse Logs". It's a solution to diagnose microservice systems using very sparse API logs. Experiments in the paper are conducted on a real commercial system, and the input data are placed in the `./data/ibm_micro_service` directory. Users can also replace it with their own data.

### Requirements
Run the following command to prepare the runnning environment.
```bash
$ python install -r requirements.txt 
```

### Experiments in the paper
To reproduce the results in our paper, just run different scripts:
- `Table 2&3`:
    - run the script `generateMask.ipynb` to control different data missing rates
    - run the script `compare.ipynb` to get the `PR@1~5` and `RankScore` results
- `Table 4& Figure 7&8`:
    - run the script `count_causal_peaks_cleaned.py` to get number of causal peaks and paths under different conditions
- `Figure 9&10&11`:
    - run the script `plot_causal.py` to get the results of `Figure 9`
    - run the script `plot_step.py` to get the results of `Figure 10`
    - run the script `plot_rate.py` to get the results of `Figure 11`

### Experiments on other data
Users can conduct experiments on their own data. To make our code easier to use, we have modularized the functions and placed them under each folder. All the functions related to our algorithms are placed in the `./algo` folder. `./methods` folder provides example usages about how to use modularized functions to get the results in 6~7 lines. For example, `MicroCU` is conducted as follows:
```python
def rcaMicroCU(cData, _head, config, display=False):
    fData = fillData(cData, config)
    grangerRes = grangerCausalmp(fData, _head, config)
    matrix, peak = buildGraph(grangerRes, config)
    mtx = genGraph(matrix, peak, config)
    nodesABC, _aInfo = analyzeRootABC(matrix, peak, cData, config)
    metricsABC = displayRes(nodesABC, config, display=display)
    return metricsABC, (nodesABC,)
```
So to conduct experiments on users' own data, they just need to make sure the data is in `numpy` format, where each row represents a particular metric, and each column represents a time point. Then simply pass it as `cData` and name each metric in `_head` and run the script. Additionally, one can reset the hyperparameters(abnormal length/step/path/c_corr) by modifying the content of `config.json` in the `./params/config.json` file.
