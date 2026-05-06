# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

if __name__ == '__main__':
    data_dir = 'data/'
    corr_stats = np.load(os.path.join(data_dir, 'correlation_stats.npy'))
    corr = corr_stats[0]
    p_val = corr_stats[1]
    fraction_anomalous = corr_stats[2]
    mean_FTLE = corr_stats[3]
    median_FTLE = corr_stats[4]
    std_FTLE = corr_stats[5]
    print('\n--- FTLE and Q-criterion Statistics ---')
    print('FTLE Mean: ' + str(round(mean_FTLE, 4)))
    print('FTLE Median: ' + str(round(median_FTLE, 4)))
    print('FTLE Standard Deviation: ' + str(round(std_FTLE, 4)))
    print('Pearson correlation (FTLE vs Q>0 residence time): ' + str(round(corr, 4)) + ' (p-value: ' + str(round(p_val, 4)) + ')')
    print('Fraction of anomalous tracers (> 2 std displacement): ' + str(round(fraction_anomalous, 4)))