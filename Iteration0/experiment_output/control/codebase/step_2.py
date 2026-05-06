# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter

if __name__ == '__main__':
    data_dir = "data/"
    traj_filepath = os.path.join(data_dir, "unwrapped_trajectories.npy")
    time_filepath = os.path.join(data_dir, "time_array.npy")
    unwrapped_trajectories = np.load(traj_filepath)
    time_array = np.load(time_filepath)
    time_lag = time_array - time_array[0]
    displacements = unwrapped_trajectories - unwrapped_trajectories[:, 0:1, :]
    sq_disp = displacements ** 2
    MSD_x = np.mean(sq_disp[:, :, 0], axis=0)
    MSD_y = np.mean(sq_disp[:, :, 1], axis=0)
    MSD_z = np.mean(sq_disp[:, :, 2], axis=0)
    MSD = MSD_x + MSD_y + MSD_z
    RMSD = np.sqrt(MSD)
    lambda_t = np.zeros_like(MSD)
    lambda_t[0] = 1.0
    max_MSD_dir = np.maximum(np.maximum(MSD_x[1:], MSD_y[1:]), MSD_z[1:])
    min_MSD_dir = np.minimum(np.minimum(MSD_x[1:], MSD_y[1:]), MSD_z[1:])
    min_MSD_dir = np.maximum(min_MSD_dir, 1e-12)
    lambda_t[1:] = max_MSD_dir / min_MSD_dir
    time_avg_lambda = np.mean(lambda_t[1:])
    print("Time-averaged anisotropy ratio (lambda): " + str(round(time_avg_lambda, 4)))
    log_t = np.log(time_lag[1:])
    log_MSD = np.log(np.maximum(MSD[1:], 1e-12))
    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_MSD)
    print("Global power-law exponent alpha: " + str(round(slope, 4)) + " +/- " + str(round(std_err, 4)))
    dlogMSD_dlogt = np.gradient(log_MSD, log_t)
    window = min(21, len(dlogMSD_dlogt) // 2 * 2 + 1)
    if window < 3:
        window = 3
    alpha_t_smooth = savgol_filter(dlogMSD_dlogt, window_length=window, polyorder=2)
    alpha_t_full = np.zeros_like(MSD)
    alpha_t_full[0] = alpha_t_smooth[0]
    alpha_t_full[1:] = alpha_t_smooth
    print("\nIdentified Scaling Regimes:")
    current_regime = None
    start_t = None
    for t, a in zip(time_lag[1:], alpha_t_smooth):
        if a >= 1.8:
            regime = 'Ballistic'
        elif 1.2 < a < 1.8:
            regime = 'Superdiffusive'
        elif 0.8 <= a <= 1.2:
            regime = 'Diffusive'
        else:
            regime = 'Subdiffusive'
        if regime != current_regime:
            if current_regime is not None:
                print("  " + current_regime + ": t = " + str(round(start_t, 3)) + " to " + str(round(t, 3)))
            current_regime = regime
            start_t = t
    if current_regime is not None:
        print("  " + current_regime + ": t = " + str(round(start_t, 3)) + " to " + str(round(time_lag[-1], 3)))
    np.save(os.path.join(data_dir, "MSD.npy"), MSD)
    np.save(os.path.join(data_dir, "MSD_x.npy"), MSD_x)
    np.save(os.path.join(data_dir, "MSD_y.npy"), MSD_y)
    np.save(os.path.join(data_dir, "MSD_z.npy"), MSD_z)
    np.save(os.path.join(data_dir, "RMSD.npy"), RMSD)
    np.save(os.path.join(data_dir, "lambda_t.npy"), lambda_t)
    np.save(os.path.join(data_dir, "alpha_t.npy"), alpha_t_full)
    print("\nSaved MSD, RMSD, directional MSDs, lambda_t, and alpha_t to data/")