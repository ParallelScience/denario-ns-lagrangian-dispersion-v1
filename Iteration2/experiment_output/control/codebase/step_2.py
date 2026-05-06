# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import pyvista as pv
import numpy as np
import gc
import time
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['text.usetex'] = False

def load_snapshot(idx):
    filepath = 'data/Turb.hydro_w.' + str(idx) + '.vtk'
    mesh = pv.read(filepath)
    vx = mesh['velx'].reshape(128, 128, 128)
    vy = mesh['vely'].reshape(128, 128, 128)
    vz = mesh['velz'].reshape(128, 128, 128)
    return vx, vy, vz

def compute_Q(vx, vy, vz, dx):
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx)
    dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dx)
    dvx_dz = (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / (2 * dx)
    dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
    dvy_dz = (np.roll(vy, -1, axis=2) - np.roll(vy, 1, axis=2)) / (2 * dx)
    dvz_dx = (np.roll(vz, -1, axis=0) - np.roll(vz, 1, axis=0)) / (2 * dx)
    dvz_dy = (np.roll(vz, -1, axis=1) - np.roll(vz, 1, axis=1)) / (2 * dx)
    dvz_dz = (np.roll(vz, -1, axis=2) - np.roll(vz, 1, axis=2)) / (2 * dx)
    S11, S22, S33 = dvx_dx, dvy_dy, dvz_dz
    S12 = 0.5 * (dvx_dy + dvy_dx)
    S13 = 0.5 * (dvx_dz + dvz_dx)
    S23 = 0.5 * (dvy_dz + dvy_dy)
    O12 = 0.5 * (dvx_dy - dvy_dx)
    O13 = 0.5 * (dvx_dz - dvz_dx)
    O23 = 0.5 * (dvy_dz - dvz_dy)
    norm_S_sq = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)
    norm_O_sq = 2*(O12**2 + O13**2 + O23**2)
    return 0.5 * (norm_O_sq - norm_S_sq)

def compute_V_LS(vx, vy, vz):
    vx_f = np.fft.fftn(vx)
    vy_f = np.fft.fftn(vy)
    vz_f = np.fft.fftn(vz)
    kx = np.fft.fftfreq(128, d=1.0/128.0)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    mask = (K >= 1) & (K <= 3)
    vx_f[~mask] = 0
    vy_f[~mask] = 0
    vz_f[~mask] = 0
    return np.fft.ifftn(vx_f).real, np.fft.ifftn(vy_f).real, np.fft.ifftn(vz_f).real

def precompute_and_cache():
    indices = list(range(18903, 19899, 5))
    for idx in indices:
        cache_file = 'data/cache_' + str(idx) + '.npz'
        if os.path.exists(cache_file):
            continue
        vx, vy, vz = load_snapshot(idx)
        Q = compute_Q(vx, vy, vz, 0.0078125)
        vx_ls, vy_ls, vz_ls = compute_V_LS(vx, vy, vz)
        V = np.stack([vx, vy, vz], axis=-1).astype(np.float32)
        V_LS = np.stack([vx_ls, vy_ls, vz_ls], axis=-1).astype(np.float32)
        np.savez(cache_file, V=V, V_LS=V_LS, Q=Q.astype(np.float32))
        del vx, vy, vz, Q, vx_ls, vy_ls, vz_ls, V, V_LS
        gc.collect()

def integrate_tracers():
    indices = list(range(18903, 19899, 5))
    num_snapshots = len(indices)
    np.random.seed(42)
    tracers = np.zeros((8000, 3), dtype=np.float64)
    tracers[:5000] = np.random.uniform(-0.5, 0.5, (5000, 3))
    dx = 0.0078125
    delta = 0.01 * dx
    tracers[5000:6000] = tracers[:1000] + np.array([delta, 0, 0])
    tracers[6000:7000] = tracers[:1000] + np.array([0, delta, 0])
    tracers[7000:8000] = tracers[:1000] + np.array([0, 0, delta])
    tracers = (tracers + 0.5) % 1.0 - 0.5
    trajectories = np.zeros((num_snapshots, 8000, 3), dtype=np.float32)
    V_tracers = np.zeros((num_snapshots, 5000, 3), dtype=np.float32)
    V_LS_tracers = np.zeros((num_snapshots, 5000, 3), dtype=np.float32)
    Q_tracers = np.zeros((num_snapshots, 5000), dtype=np.float32)
    x_pad = np.linspace(-0.5 - dx/2, 0.5 + dx/2, 130)
    dt = 0.05 / 10
    for i, idx in enumerate(indices):
        data = np.load('data/cache_' + str(idx) + '.npz')
        V, V_LS, Q = data['V'], data['V_LS'], data['Q']
        V_pad = np.pad(V, pad_width=((1,1), (1,1), (1,1), (0,0)), mode='wrap')
        V_LS_pad = np.pad(V_LS, pad_width=((1,1), (1,1), (1,1), (0,0)), mode='wrap')
        Q_pad = np.pad(Q, pad_width=1, mode='wrap')
        interp_V = RegularGridInterpolator((x_pad, x_pad, x_pad), V_pad, method='linear')
        interp_V_LS = RegularGridInterpolator((x_pad, x_pad, x_pad), V_LS_pad, method='linear')
        interp_Q = RegularGridInterpolator((x_pad, x_pad, x_pad), Q_pad, method='linear')
        trajectories[i] = tracers
        V_tracers[i] = interp_V(tracers[:5000])
        V_LS_tracers[i] = interp_V_LS(tracers[:5000])
        Q_tracers[i] = interp_Q(tracers[:5000])
        if i < num_snapshots - 1:
            for _ in range(10):
                k1 = dt * interp_V(tracers)
                k2 = dt * interp_V((tracers + k1/2 + 0.5) % 1.0 - 0.5)
                k3 = dt * interp_V((tracers + k2/2 + 0.5) % 1.0 - 0.5)
                k4 = dt * interp_V((tracers + k3 + 0.5) % 1.0 - 0.5)
                tracers = (tracers + (k1 + 2*k2 + 2*k3 + k4) / 6 + 0.5) % 1.0 - 0.5
        if i == num_snapshots - 1:
            plt.figure(figsize=(10, 8))
            plt.pcolormesh(np.linspace(-0.5, 0.5, 128), np.linspace(-0.5, 0.5, 128), Q[:, :, 64].T, shading='nearest', cmap='viridis')
            plt.colorbar(label='Q-criterion')
            plt.title('Q-criterion slice at z=0, final snapshot')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plot_filename = 'data/Q_criterion_slice_1_' + str(int(time.time())) + '.png'
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print('Plot saved to ' + plot_filename)
    np.savez('data/tracer_data.npz', trajectories=trajectories, V_tracers=V_tracers, V_LS_tracers=V_LS_tracers, Q_tracers=Q_tracers)

if __name__ == '__main__':
    precompute_and_cache()
    integrate_tracers()