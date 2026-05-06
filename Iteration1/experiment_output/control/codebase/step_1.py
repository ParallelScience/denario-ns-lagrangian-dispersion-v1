# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import time
import requests
import concurrent.futures
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import datetime
plt.rcParams['text.usetex'] = False
def download_file(url, dest_path, retries=3, delay=2):
    if os.path.exists(dest_path):
        return True
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print('Failed to download ' + url + ': ' + str(e))
                return False
def download_snapshots(indices, data_dir):
    urls = ['https://huggingface.co/datasets/pedrota2000/NS_simulation/resolve/main/Turb.hydro_w.' + str(idx) + '.vtk' for idx in indices]
    dest_paths = [os.path.join(data_dir, 'Turb.hydro_w.' + str(idx) + '.vtk') for idx in indices]
    print('Starting download of ' + str(len(indices)) + ' snapshots...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(download_file, urls, dest_paths))
    success_count = sum(results)
    print('Successfully downloaded/verified ' + str(success_count) + ' out of ' + str(len(indices)) + ' files.')
def read_vtk(filepath):
    mesh = pv.read(filepath)
    velx = mesh['velx'].reshape(128, 128, 128)
    vely = mesh['vely'].reshape(128, 128, 128)
    velz = mesh['velz'].reshape(128, 128, 128)
    return velx, vely, velz
def compute_velocity_gradient(velx, vely, velz, dx):
    grad_v = np.zeros((3, 3, 128, 128, 128), dtype=np.float32)
    vels = [velx, vely, velz]
    for i in range(3):
        for j in range(3):
            grad_v[i, j] = (np.roll(vels[i], -1, axis=j) - np.roll(vels[i], 1, axis=j)) / (2 * dx)
    return grad_v
def compute_vorticity(grad_v):
    omega_x = grad_v[2, 1] - grad_v[1, 2]
    omega_y = grad_v[0, 2] - grad_v[2, 0]
    omega_z = grad_v[1, 0] - grad_v[0, 1]
    return omega_x, omega_y, omega_z
def compute_q_criterion(grad_v):
    S = 0.5 * (grad_v + np.transpose(grad_v, (1, 0, 2, 3, 4)))
    Omega = 0.5 * (grad_v - np.transpose(grad_v, (1, 0, 2, 3, 4)))
    norm_Omega_sq = np.sum(Omega**2, axis=(0, 1))
    norm_S_sq = np.sum(S**2, axis=(0, 1))
    Q = 0.5 * (norm_Omega_sq - norm_S_sq)
    return Q
def compute_large_scale_velocity(velx, vely, velz, n_max=3):
    k = np.fft.fftfreq(128) * 128
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    n = np.sqrt(kx**2 + ky**2 + kz**2)
    mask = n <= n_max
    velx_ls = np.real(np.fft.ifftn(np.fft.fftn(velx) * mask))
    vely_ls = np.real(np.fft.ifftn(np.fft.fftn(vely) * mask))
    velz_ls = np.real(np.fft.ifftn(np.fft.fftn(velz) * mask))
    return velx_ls, vely_ls, velz_ls
if __name__ == '__main__':
    data_dir = '/home/node/work/projects/ns_lagrangian_v1/data/'
    output_dir = 'data/'
    indices = list(range(18903, 19898 + 1, 5))
    download_snapshots(indices, data_dir)
    first_snapshot_path = os.path.join(data_dir, 'Turb.hydro_w.18903.vtk')
    print('Reading first snapshot: ' + first_snapshot_path)
    velx, vely, velz = read_vtk(first_snapshot_path)
    dx = 0.0078125
    print('Computing velocity gradient...')
    grad_v = compute_velocity_gradient(velx, vely, velz, dx)
    print('Computing vorticity...')
    omega_x, omega_y, omega_z = compute_vorticity(grad_v)
    vort_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    print('Computing Q-criterion...')
    Q = compute_q_criterion(grad_v)
    print('Computing large-scale velocity...')
    velx_ls, vely_ls, velz_ls = compute_large_scale_velocity(velx, vely, velz, n_max=3)
    vel_mag = np.sqrt(velx**2 + vely**2 + velz**2)
    vel_rms = np.sqrt(np.mean(vel_mag**2))
    print('\n--- First Snapshot Statistics ---')
    print('Velocity RMS: ' + str(np.round(vel_rms, 4)))
    print('Q-criterion - Min: ' + str(np.round(np.min(Q), 4)) + ', Max: ' + str(np.round(np.max(Q), 4)) + ', Mean: ' + str(np.round(np.mean(Q), 4)) + ', Std: ' + str(np.round(np.std(Q), 4)))
    print('Vorticity Magnitude - Min: ' + str(np.round(np.min(vort_mag), 4)) + ', Max: ' + str(np.round(np.max(vort_mag), 4)) + ', Mean: ' + str(np.round(np.mean(vort_mag), 4)) + ', Std: ' + str(np.round(np.std(vort_mag), 4)))
    print('---------------------------------\n')
    print('Generating plot...')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    slice_idx = 64
    Q_slice = Q[:, :, slice_idx]
    vort_slice = vort_mag[:, :, slice_idx]
    im1 = axes[0].imshow(Q_slice.T, origin='lower', cmap='RdBu_r', vmin=-np.std(Q_slice)*3, vmax=np.std(Q_slice)*3, extent=[-0.5, 0.5, -0.5, 0.5])
    axes[0].set_title('Q-criterion (z=0)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im1, ax=axes[0], label='Q')
    im2 = axes[1].imshow(vort_slice.T, origin='lower', cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5])
    axes[1].set_title('Vorticity Magnitude (z=0)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    fig.colorbar(im2, ax=axes[1], label='|ω|')
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = 'q_vorticity_slice_1_' + timestamp + '.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print('Plot saved to ' + filepath)