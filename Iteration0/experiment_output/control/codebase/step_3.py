# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.stats import kurtosis, linregress
from scipy.integrate import trapezoid

def compute_vacf(velocities):
    """
    Computes the Velocity Autocorrelation Function (VACF) averaged over all tracers.
    
    Parameters:
    velocities (np.ndarray): Array of shape (N_tracers, N_time_steps, 3) containing tracer velocities.
    
    Returns:
    np.ndarray: 1D array of length N_time_steps containing the VACF.
    """
    N_v = velocities.shape[1]
    vacf = np.zeros(N_v)
    for k in range(N_v):
        if k == 0:
            vacf[k] = np.mean(np.sum(velocities * velocities, axis=2))
        else:
            vacf[k] = np.mean(np.sum(velocities[:, :-k, :] * velocities[:, k:, :], axis=2))
    return vacf

if __name__ == '__main__':
    data_dir = "data/"
    traj_filepath = os.path.join(data_dir, "unwrapped_trajectories.npy")
    time_filepath = os.path.join(data_dir, "time_array.npy")
    unwrapped_trajectories = np.load(traj_filepath)
    time_array = np.load(time_filepath)
    N_tracers, N_snapshots, _ = unwrapped_trajectories.shape
    idx1 = int(N_snapshots * 0.1)
    idx2 = int(N_snapshots * 0.3)
    idx3 = int(N_snapshots * 0.6)
    idx4 = N_snapshots - 1
    indices = [idx1, idx2, idx3, idx4]
    lag_times = time_array[indices] - time_array[0]
    pdfs = []
    bin_centers_list = []
    tail_exponents = []
    print("--- Displacement Statistics ---")
    for i, idx in enumerate(indices):
        disp = unwrapped_trajectories[:, idx, :] - unwrapped_trajectories[:, 0, :]
        dist = np.linalg.norm(disp, axis=1)
        kurt_r = kurtosis(dist, fisher=True)
        kurt_x = kurtosis(disp[:, 0], fisher=True)
        kurt_y = kurtosis(disp[:, 1], fisher=True)
        kurt_z = kurtosis(disp[:, 2], fisher=True)
        kurt_1d_avg = (kurt_x + kurt_y + kurt_z) / 3.0
        print("Lag time " + str(round(lag_times[i], 3)) + " (idx " + str(idx) + "):")
        print("  Excess Kurtosis (1D components avg): " + str(round(kurt_1d_avg, 4)))
        print("  Excess Kurtosis (magnitude r): " + str(round(kurt_r, 4)))
        min_val = np.min(dist[dist > 0])
        if min_val == 0 or np.isnan(min_val):
            min_val = 1e-6
        max_val = np.max(dist)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
        hist, bin_edges = np.histogram(dist, bins=bins, density=True)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        threshold = np.percentile(dist, 90)
        mask = (bin_centers > threshold) & (hist > 0)
        if np.sum(mask) > 2:
            log_x = np.log(bin_centers[mask])
            log_y = np.log(hist[mask])
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
            tail_exp = slope
            alpha_L = -1.0 - tail_exp
        else:
            tail_exp = np.nan
            alpha_L = np.nan
        print("  Tail power-law exponent (slope): " + str(round(tail_exp, 4)))
        print("  Estimated Levy stability index alpha_L: " + str(round(alpha_L, 4)))
        pdfs.append(hist)
        bin_centers_list.append(bin_centers)
        tail_exponents.append(tail_exp)
    dt = time_array[1:] - time_array[:-1]
    velocities = np.diff(unwrapped_trajectories, axis=1) / dt[np.newaxis, :, np.newaxis]
    dt_mean = np.mean(dt)
    print("\n--- Velocity Autocorrelation Function ---")
    vacf = compute_vacf(velocities)
    vacf_norm = vacf / vacf[0]
    integral_unnorm = trapezoid(vacf, dx=dt_mean)
    taylor_diff_coeff = integral_unnorm / 3.0
    integral_norm = trapezoid(vacf_norm, dx=dt_mean)
    print("VACF(0) [Mean squared velocity]: " + str(round(vacf[0], 4)))
    print("Time integral of normalized VACF: " + str(round(integral_norm, 4)))
    print("Taylor diffusion coefficient (1/3 * integral of unnormalized VACF): " + str(round(taylor_diff_coeff, 4)))
    np.save(os.path.join(data_dir, "pdfs.npy"), np.array(pdfs))
    np.save(os.path.join(data_dir, "bin_centers.npy"), np.array(bin_centers_list))
    np.save(os.path.join(data_dir, "tail_exponents.npy"), np.array(tail_exponents))
    np.save(os.path.join(data_dir, "vacf_norm.npy"), vacf_norm)
    np.save(os.path.join(data_dir, "vacf.npy"), vacf)
    np.save(os.path.join(data_dir, "lag_times_pdf.npy"), np.array(lag_times))
    print("\nSaved PDFs, bin centers, tail exponents, and VACF arrays to data/")