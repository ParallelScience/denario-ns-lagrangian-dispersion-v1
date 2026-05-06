# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import urllib.request
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator

def download_snapshots(indices, data_dir="data/"):
    base_url_prefix = "https://huggingface.co/datasets/pedrota2000/NS_simulation/resolve/main/Turb.hydro_w."
    base_url_suffix = ".vtk"
    filepaths = []
    for idx in indices:
        idx_str = str(idx).zfill(5)
        filename = "Turb.hydro_w." + idx_str + ".vtk"
        filepath = os.path.join(data_dir, filename)
        filepaths.append(filepath)
        if not os.path.exists(filepath):
            url = base_url_prefix + idx_str + base_url_suffix
            print("Downloading " + filename + "...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print("Error downloading " + filename + ": " + str(e))
                raise e
    return filepaths

def load_velocity_interpolator(filepath):
    mesh = pv.read(filepath)
    velx = mesh['velx'].reshape(128, 128, 128)
    vely = mesh['vely'].reshape(128, 128, 128)
    velz = mesh['velz'].reshape(128, 128, 128)
    dx = 1.0 / 128.0
    x = np.linspace(-0.5 + dx/2.0, 0.5 - dx/2.0, 128)
    x_pad = np.concatenate([[x[0] - dx], x, [x[-1] + dx]])
    velx_pad = np.pad(velx, pad_width=1, mode='wrap')
    vely_pad = np.pad(vely, pad_width=1, mode='wrap')
    velz_pad = np.pad(velz, pad_width=1, mode='wrap')
    vel_pad = np.stack([velx_pad, vely_pad, velz_pad], axis=-1)
    interp = RegularGridInterpolator((x_pad, x_pad, x_pad), vel_pad, bounds_error=False, fill_value=None)
    rms_vel = np.sqrt(np.mean(velx**2 + vely**2 + velz**2))
    return interp, rms_vel

def get_velocity(r, t, t_n, dt_snapshot, interp_n, interp_np1):
    tau = np.clip((t - t_n) / dt_snapshot, 0.0, 1.0)
    r_wrapped = (r + 0.5) % 1.0 - 0.5
    v_n = interp_n(r_wrapped)
    v_np1 = interp_np1(r_wrapped)
    return (1.0 - tau) * v_n + tau * v_np1

def rk4_step(r, t, dt, t_n, dt_snapshot, interp_n, interp_np1):
    k1 = get_velocity(r, t, t_n, dt_snapshot, interp_n, interp_np1)
    k2 = get_velocity(r + k1 * dt / 2.0, t + dt / 2.0, t_n, dt_snapshot, interp_n, interp_np1)
    k3 = get_velocity(r + k2 * dt / 2.0, t + dt / 2.0, t_n, dt_snapshot, interp_n, interp_np1)
    k4 = get_velocity(r + k3 * dt, t + dt, t_n, dt_snapshot, interp_n, interp_np1)
    return r + (k1 + 2.0*k2 + 2.0*k3 + k4) * dt / 6.0

def minimum_image(dr, L=1.0):
    return dr - np.round(dr / L) * L

if __name__ == '__main__':
    data_dir = "data/"
    indices = list(range(18903, 19903, 5))
    if len(indices) > 200:
        indices = indices[:200]
    print("Starting download of " + str(len(indices)) + " snapshots...")
    filepaths = download_snapshots(indices, data_dir)
    print("All snapshots downloaded or already present.")
    N_tracers = 10000
    N_snapshots = len(indices)
    sub_steps = 5
    dt_snapshot = 0.05
    dt = dt_snapshot / sub_steps
    np.random.seed(42)
    pos = np.random.uniform(-0.5, 0.5, (N_tracers, 3))
    unwrapped_pos = pos.copy()
    unwrapped_trajectories = np.zeros((N_tracers, N_snapshots, 3))
    unwrapped_trajectories[:, 0, :] = unwrapped_pos
    time_array = np.zeros(N_snapshots)
    time_array[0] = indices[0] * 0.01
    has_crossed = np.zeros(N_tracers, dtype=bool)
    rms_velocities = []
    print("Loading initial snapshot...")
    interp_n, rms_n = load_velocity_interpolator(filepaths[0])
    rms_velocities.append(rms_n)
    start_time = time.time()
    print("Starting advection loop...")
    for n in range(N_snapshots - 1):
        if (n + 1) % 10 == 0:
            print("Processing interval " + str(n + 1) + " / " + str(N_snapshots - 1))
        interp_np1, rms_np1 = load_velocity_interpolator(filepaths[n+1])
        rms_velocities.append(rms_np1)
        t_n = indices[n] * 0.01
        for s in range(sub_steps):
            t_current = t_n + s * dt
            pos_new_unwrapped = rk4_step(pos, t_current, dt, t_n, dt_snapshot, interp_n, interp_np1)
            pos_new = (pos_new_unwrapped + 0.5) % 1.0 - 0.5
            dr = pos_new - pos
            dr_min = minimum_image(dr, L=1.0)
            unwrapped_pos += dr_min
            crossed_this_step = np.any(np.abs(np.round(dr / 1.0)) > 0, axis=1)
            has_crossed |= crossed_this_step
            pos = pos_new
        unwrapped_trajectories[:, n+1, :] = unwrapped_pos
        time_array[n+1] = indices[n+1] * 0.01
        interp_n = interp_np1
    end_time = time.time()
    traj_filepath = os.path.join(data_dir, "unwrapped_trajectories.npy")
    time_filepath = os.path.join(data_dir, "time_array.npy")
    np.save(traj_filepath, unwrapped_trajectories)
    np.save(time_filepath, time_array)
    print("Trajectories saved to " + traj_filepath)
    print("Time array saved to " + time_filepath)
    avg_rms_vel = np.mean(rms_velocities)
    fraction_crossed = np.mean(has_crossed)
    final_displacements = unwrapped_trajectories[:, -1, :] - unwrapped_trajectories[:, 0, :]
    final_distances = np.linalg.norm(final_displacements, axis=1)
    min_disp = np.min(final_distances)
    max_disp = np.max(final_distances)
    print("\n--- Diagnostics ---")
    print("Average RMS velocity used: " + str(avg_rms_vel))
    print("Fraction of tracers that crossed periodic boundaries: " + str(fraction_crossed))
    print("Minimum unwrapped displacement at final time: " + str(min_disp))
    print("Maximum unwrapped displacement at final time: " + str(max_disp))
    print("Wall-clock time taken for advection: " + str(end_time - start_time) + " seconds")