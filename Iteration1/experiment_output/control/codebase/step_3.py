# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np

def compute_msd_and_anisotropy():
    data_dir = 'data/'
    print('Loading data...')
    unwrapped_trajectories = np.load(os.path.join(data_dir, 'unwrapped_trajectories.npy'))
    sampled_Q = np.load(os.path.join(data_dir, 'sampled_Q.npy'))
    initial_vel_ls = np.load(os.path.join(data_dir, 'initial_vel_ls.npy'))
    N_snapshots, N_total, _ = unwrapped_trajectories.shape
    N_primary = 5000
    dt_snap = 0.05
    t = np.arange(N_snapshots) * dt_snap
    print('Computing Ensemble MSD...')
    disp = unwrapped_trajectories[:, :N_primary, :] - unwrapped_trajectories[0, :N_primary, :]
    msd_ensemble = np.mean(np.sum(disp**2, axis=2), axis=1)
    print('Computing alpha(t)...')
    t_valid = t[1:]
    msd_valid = msd_ensemble[1:]
    log_t = np.log(t_valid)
    log_msd = np.log(msd_valid)
    alpha_t = np.gradient(log_msd, log_t)
    p_global = np.polyfit(log_t, log_msd, 1)
    alpha_global = p_global[0]
    early_mask = t_valid < 1.0
    if np.any(early_mask):
        p_early = np.polyfit(log_t[early_mask], log_msd[early_mask], 1)
        alpha_early = p_early[0]
    else:
        alpha_early = np.nan
    late_mask = t_valid > 5.0
    if np.any(late_mask):
        p_late = np.polyfit(log_t[late_mask], log_msd[late_mask], 1)
        alpha_late = p_late[0]
    else:
        alpha_late = np.nan
    print('Computing Trapped and Free cohorts...')
    is_high_Q = sampled_Q > 0
    residence_time = np.sum(is_high_Q, axis=0) * dt_snap
    p80 = np.percentile(residence_time, 80)
    p20 = np.percentile(residence_time, 20)
    trapped_idx = np.where(residence_time >= p80)[0]
    free_idx = np.where(residence_time <= p20)[0]
    msd_trapped = np.mean(np.sum(disp[:, trapped_idx, :]**2, axis=2), axis=1)
    msd_free = np.mean(np.sum(disp[:, free_idx, :]**2, axis=2), axis=1)
    print('Computing Boundary-Crossers and Non-Crossers...')
    crossers_mask = np.any(np.abs(unwrapped_trajectories[:, :N_primary, :]) > 0.5, axis=(0, 2))
    crossers_idx = np.where(crossers_mask)[0]
    non_crossers_idx = np.where(~crossers_mask)[0]
    if len(crossers_idx) > 0:
        msd_crossers = np.mean(np.sum(disp[:, crossers_idx, :]**2, axis=2), axis=1)
    else:
        msd_crossers = np.zeros_like(msd_ensemble)
    if len(non_crossers_idx) > 0:
        msd_non_crossers = np.mean(np.sum(disp[:, non_crossers_idx, :]**2, axis=2), axis=1)
    else:
        msd_non_crossers = np.zeros_like(msd_ensemble)
    fraction_crossers = len(crossers_idx) / N_primary
    print('Computing Anisotropy...')
    v0 = initial_vel_ls
    v0_norm = np.linalg.norm(v0, axis=1, keepdims=True)
    v0_hat = v0 / (v0_norm + 1e-12)
    disp_para = np.sum(disp * v0_hat[np.newaxis, :, :], axis=2)
    disp_perp_sq = np.sum(disp**2, axis=2) - disp_para**2
    msd_para = np.mean(disp_para**2, axis=1)
    msd_perp = np.mean(disp_perp_sq, axis=1) / 2.0
    lambda_t = np.zeros_like(msd_para)
    valid_idx = msd_perp > 1e-12
    lambda_t[valid_idx] = msd_para[valid_idx] / msd_perp[valid_idx]
    lambda_t[~valid_idx] = 1.0
    mean_lambda = np.mean(lambda_t[1:])
    std_lambda = np.std(lambda_t[1:])
    print('Computing VACF and Taylor diffusion...')
    v_tracer = np.diff(unwrapped_trajectories[:, :N_primary, :], axis=0) / dt_snap
    N_v = len(v_tracer)
    vacf = np.zeros(N_v)
    for tau in range(N_v):
        vacf[tau] = np.mean(np.sum(v_tracer[:N_v-tau] * v_tracer[tau:], axis=2))
    taylor_diff = np.cumsum(vacf) * dt_snap
    print('\n--- Key Scalar Results ---')
    print('Fitted global power-law exponent alpha: ' + str(np.round(alpha_global, 4)))
    print('Early-time alpha (t < 1.0): ' + str(np.round(alpha_early, 4)))
    print('Late-time alpha (t > 5.0): ' + str(np.round(alpha_late, 4)))
    print('Mean anisotropy ratio lambda: ' + str(np.round(mean_lambda, 4)) + ' +/- ' + str(np.round(std_lambda, 4)))
    print('Fraction of Boundary-Crossers: ' + str(np.round(fraction_crossers * 100, 2)) + '%')
    print('--------------------------\n')
    print('Saving results...')
    np.savez(os.path.join(data_dir, 'step_3_results.npz'), t=t, msd_ensemble=msd_ensemble, msd_trapped=msd_trapped, msd_free=msd_free, msd_crossers=msd_crossers, msd_non_crossers=msd_non_crossers, alpha_t=alpha_t, lambda_t=lambda_t, vacf=vacf, taylor_diff=taylor_diff)
    print('Done.')

if __name__ == '__main__':
    compute_msd_and_anisotropy()