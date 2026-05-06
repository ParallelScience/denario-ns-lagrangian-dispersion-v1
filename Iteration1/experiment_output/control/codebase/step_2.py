# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
from scipy.ndimage import map_coordinates
from step_1 import read_vtk, compute_velocity_gradient, compute_q_criterion, compute_large_scale_velocity

def get_v(x, alpha, vx1, vy1, vz1, vx2, vy2, vz2):
    idx = (x + 0.5) * 128.0 - 0.5
    idx = idx.T
    v1x = map_coordinates(vx1, idx, mode='wrap', order=1)
    v1y = map_coordinates(vy1, idx, mode='wrap', order=1)
    v1z = map_coordinates(vz1, idx, mode='wrap', order=1)
    v2x = map_coordinates(vx2, idx, mode='wrap', order=1)
    v2y = map_coordinates(vy2, idx, mode='wrap', order=1)
    v2z = map_coordinates(vz2, idx, mode='wrap', order=1)
    vx = (1.0 - alpha) * v1x + alpha * v2x
    vy = (1.0 - alpha) * v1y + alpha * v2y
    vz = (1.0 - alpha) * v1z + alpha * v2z
    return np.stack([vx, vy, vz], axis=-1)

if __name__ == '__main__':
    data_dir = '/home/node/work/projects/ns_lagrangian_v1/data/'
    output_dir = 'data/'
    indices = list(range(18903, 19898 + 1, 5))
    N_snapshots = len(indices)
    N_primary = 5000
    N_aux = 3000
    N_total = N_primary + N_aux
    unwrapped_trajectories = np.zeros((N_snapshots, N_total, 3), dtype=np.float32)
    wrapped_trajectories = np.zeros((N_snapshots, N_total, 3), dtype=np.float32)
    sampled_Q = np.zeros((N_snapshots, N_primary), dtype=np.float32)
    sampled_vel_ls = np.zeros((N_snapshots, N_primary, 3), dtype=np.float32)
    np.random.seed(42)
    x = np.random.uniform(-0.5, 0.5, (N_total, 3)).astype(np.float32)
    eps = 1e-5
    x[5000:6000] = x[0:1000] + np.array([eps, 0, 0], dtype=np.float32)
    x[6000:7000] = x[0:1000] + np.array([0, eps, 0], dtype=np.float32)
    x[7000:8000] = x[0:1000] + np.array([0, 0, eps], dtype=np.float32)
    x = (x + 0.5) % 1.0 - 0.5
    x_unwrapped = x.copy()
    dx = 0.0078125
    dt_snap = 0.05
    dt_sub = dt_snap / 10.0
    velx1, vely1, velz1 = read_vtk(os.path.join(data_dir, 'Turb.hydro_w.' + str(indices[0]) + '.vtk'))
    grad_v1 = compute_velocity_gradient(velx1, vely1, velz1, dx)
    Q1 = compute_q_criterion(grad_v1)
    velx_ls1, vely_ls1, velz_ls1 = compute_large_scale_velocity(velx1, vely1, velz1)
    unwrapped_trajectories[0] = x_unwrapped
    wrapped_trajectories[0] = x
    idx_primary = (x[:N_primary] + 0.5) * 128.0 - 0.5
    idx_primary = idx_primary.T
    sampled_Q[0] = map_coordinates(Q1, idx_primary, mode='wrap', order=1)
    sampled_vel_ls[0, :, 0] = map_coordinates(velx_ls1, idx_primary, mode='wrap', order=1)
    sampled_vel_ls[0, :, 1] = map_coordinates(vely_ls1, idx_primary, mode='wrap', order=1)
    sampled_vel_ls[0, :, 2] = map_coordinates(velz_ls1, idx_primary, mode='wrap', order=1)
    initial_vel_ls = sampled_vel_ls[0].copy()
    for i in range(N_snapshots - 1):
        velx2, vely2, velz2 = read_vtk(os.path.join(data_dir, 'Turb.hydro_w.' + str(indices[i+1]) + '.vtk'))
        grad_v2 = compute_velocity_gradient(velx2, vely2, velz2, dx)
        Q2 = compute_q_criterion(grad_v2)
        velx_ls2, vely_ls2, velz_ls2 = compute_large_scale_velocity(velx2, vely2, velz2)
        for step in range(10):
            alpha1 = step / 10.0
            alpha2 = (step + 0.5) / 10.0
            alpha3 = (step + 1.0) / 10.0
            k1 = get_v(x, alpha1, velx1, vely1, velz1, velx2, vely2, velz2)
            x_k2 = (x + k1 * dt_sub / 2.0 + 0.5) % 1.0 - 0.5
            k2 = get_v(x_k2, alpha2, velx1, vely1, velz1, velx2, vely2, velz2)
            x_k3 = (x + k2 * dt_sub / 2.0 + 0.5) % 1.0 - 0.5
            k3 = get_v(x_k3, alpha2, velx1, vely1, velz1, velx2, vely2, velz2)
            x_k4 = (x + k3 * dt_sub + 0.5) % 1.0 - 0.5
            k4 = get_v(x_k4, alpha3, velx1, vely1, velz1, velx2, vely2, velz2)
            dx_step = (k1 + 2.0*k2 + 2.0*k3 + k4) * dt_sub / 6.0
            x_unwrapped += dx_step
            x = (x + dx_step + 0.5) % 1.0 - 0.5
        unwrapped_trajectories[i+1] = x_unwrapped
        wrapped_trajectories[i+1] = x
        idx_primary = (x[:N_primary] + 0.5) * 128.0 - 0.5
        idx_primary = idx_primary.T
        sampled_Q[i+1] = map_coordinates(Q2, idx_primary, mode='wrap', order=1)
        sampled_vel_ls[i+1, :, 0] = map_coordinates(velx_ls2, idx_primary, mode='wrap', order=1)
        sampled_vel_ls[i+1, :, 1] = map_coordinates(vely_ls2, idx_primary, mode='wrap', order=1)
        sampled_vel_ls[i+1, :, 2] = map_coordinates(velz_ls2, idx_primary, mode='wrap', order=1)
        velx1, vely1, velz1 = velx2, vely2, velz2
        Q1 = Q2
        velx_ls1, vely_ls1, velz_ls1 = velx_ls2, vely_ls2, velz_ls2
    np.save(os.path.join(output_dir, 'unwrapped_trajectories.npy'), unwrapped_trajectories)
    np.save(os.path.join(output_dir, 'wrapped_trajectories.npy'), wrapped_trajectories)
    np.save(os.path.join(output_dir, 'sampled_Q.npy'), sampled_Q)
    np.save(os.path.join(output_dir, 'sampled_vel_ls.npy'), sampled_vel_ls)
    np.save(os.path.join(output_dir, 'initial_vel_ls.npy'), initial_vel_ls)