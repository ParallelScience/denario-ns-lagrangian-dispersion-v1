# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy import stats

def compute_vortex_trapping_and_ftle():
    data_dir = 'data/'
    print('Loading data...')
    unwrapped_trajectories = np.load(os.path.join(data_dir, 'unwrapped_trajectories.npy'))
    sampled_Q = np.load(os.path.join(data_dir, 'sampled_Q.npy'))
    N_snapshots = unwrapped_trajectories.shape[0]
    dt_snap = 0.05
    t = np.arange(N_snapshots) * dt_snap
    print('Computing Q-criterion autocorrelation...')
    Q_global_mean = np.mean(sampled_Q)
    Q_fluct = sampled_Q - Q_global_mean
    Q_var_mean = np.mean(Q_fluct**2)
    R_tau = np.zeros(N_snapshots)
    for tau in range(N_snapshots):
        if tau == 0:
            cov = np.mean(Q_fluct * Q_fluct)
        else:
            cov = np.mean(Q_fluct[:-tau] * Q_fluct[tau:])
        R_tau[tau] = cov / (Q_var_mean + 1e-12)
    threshold = 1.0 / np.exp(1.0)
    decorr_time = np.nan
    for i in range(1, N_snapshots):
        if R_tau[i] < threshold:
            t1, t2 = t[i-1], t[i]
            r1, r2 = R_tau[i-1], R_tau[i]
            decorr_time = t1 + (threshold - r1) * (t2 - t1) / (r2 - r1)
            break
    print('\n--- Q-criterion Autocorrelation ---')
    if np.isnan(decorr_time):
        print('Q-criterion decorrelation time: > ' + str(np.round(t[-1], 4)) + ' time units (did not drop below 1/e)')
        print('Eddy turnover time estimate: ~2.6 time units')
        print('Ratio (Decorr / Eddy): > ' + str(np.round(t[-1] / 2.6, 4)))
    else:
        print('Q-criterion decorrelation time: ' + str(np.round(decorr_time, 4)) + ' time units')
        print('Eddy turnover time estimate: ~2.6 time units')
        print('Ratio (Decorr / Eddy): ' + str(np.round(decorr_time / 2.6, 4)))
    print('-----------------------------------\n')
    print('Computing FTLE for 1,000 tracers...')
    N_ftle = 1000
    eps = 1e-5
    primary = unwrapped_trajectories[:, :N_ftle, :]
    neigh_x = unwrapped_trajectories[:, 5000:6000, :]
    neigh_y = unwrapped_trajectories[:, 6000:7000, :]
    neigh_z = unwrapped_trajectories[:, 7000:8000, :]
    F = np.zeros((N_snapshots, N_ftle, 3, 3))
    F[:, :, :, 0] = (neigh_x - primary) / eps
    F[:, :, :, 1] = (neigh_y - primary) / eps
    F[:, :, :, 2] = (neigh_z - primary) / eps
    FTLE = np.zeros((N_snapshots, N_ftle))
    for i in range(1, N_snapshots):
        F_i = F[i]
        C = np.matmul(F_i.transpose(0, 2, 1), F_i)
        eigvals = np.linalg.eigvalsh(C)
        max_eig = np.maximum(np.max(eigvals, axis=1), 1e-12)
        FTLE[i] = 1.0 / (2.0 * t[i]) * np.log(max_eig)
    print('Defining Trapped and Free cohorts for the 1,000 tracers...')
    sampled_Q_1000 = sampled_Q[:, :N_ftle]
    is_high_Q = sampled_Q_1000 > 0
    residence_time = np.sum(is_high_Q, axis=0) * dt_snap
    p80 = np.percentile(residence_time, 80)
    p20 = np.percentile(residence_time, 20)
    trapped_idx = np.where(residence_time >= p80)[0]
    free_idx = np.where(residence_time <= p20)[0]
    ftle_final = FTLE[-1]
    ftle_trapped = ftle_final[trapped_idx]
    ftle_free = ftle_final[free_idx]
    mean_ftle_trapped = np.mean(ftle_trapped)
    std_ftle_trapped = np.std(ftle_trapped)
    mean_ftle_free = np.mean(ftle_free)
    std_ftle_free = np.std(ftle_free)
    print('\n--- FTLE Statistics (Final Time) ---')
    print('Trapped Cohort FTLE - Mean: ' + str(np.round(mean_ftle_trapped, 4)) + ', Std: ' + str(np.round(std_ftle_trapped, 4)))
    print('Free Cohort FTLE    - Mean: ' + str(np.round(mean_ftle_free, 4)) + ', Std: ' + str(np.round(std_ftle_free, 4)))
    t_stat, p_val = stats.ttest_ind(ftle_trapped, ftle_free, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(ftle_trapped, ftle_free)
    print('\nStatistical Tests (Trapped vs Free):')
    print('Welch\'s t-test: t-statistic = ' + str(np.round(t_stat, 4)) + ', p-value = ' + str(p_val))
    print('Mann-Whitney U test: U-statistic = ' + str(np.round(u_stat, 4)) + ', p-value = ' + str(p_val_u))
    print('------------------------------------\n')
    print('Saving results...')
    np.savez(os.path.join(data_dir, 'step_4_results.npz'), t=t, R_tau=R_tau, FTLE=FTLE, trapped_idx_1000=trapped_idx, free_idx_1000=free_idx)
    print('Done.')

if __name__ == '__main__':
    compute_vortex_trapping_and_ftle()