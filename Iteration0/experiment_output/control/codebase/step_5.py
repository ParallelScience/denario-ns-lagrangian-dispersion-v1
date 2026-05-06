# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    timestamp = int(time.time())
    time_array = np.load(os.path.join(data_dir, 'time_array.npy'))
    time_lag = time_array - time_array[0]
    MSD = np.load(os.path.join(data_dir, 'MSD.npy'))
    MSD_x = np.load(os.path.join(data_dir, 'MSD_x.npy'))
    MSD_y = np.load(os.path.join(data_dir, 'MSD_y.npy'))
    MSD_z = np.load(os.path.join(data_dir, 'MSD_z.npy'))
    plt.figure(figsize=(8, 6))
    plt.loglog(time_lag[1:], MSD[1:], label='Total MSD', color='black', linewidth=2)
    plt.loglog(time_lag[1:], MSD_x[1:], label='MSD_x', linestyle='--')
    plt.loglog(time_lag[1:], MSD_y[1:], label='MSD_y', linestyle='--')
    plt.loglog(time_lag[1:], MSD_z[1:], label='MSD_z', linestyle='--')
    t_ref = time_lag[1:]
    plt.loglog(t_ref, MSD[1] * (t_ref / t_ref[0])**1, 'k:', label='alpha=1 (Brownian)')
    plt.loglog(t_ref, MSD[1] * (t_ref / t_ref[0])**2, 'k-.', label='alpha=2 (Ballistic)')
    plt.xlabel('Lag Time [simulation units]')
    plt.ylabel('Mean-Square Displacement (MSD) [domain units^2]')
    plt.title('Lagrangian Tracer Mean-Square Displacement')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    p1_path = os.path.join(data_dir, 'MSD_plot_1_' + str(timestamp) + '.png')
    plt.savefig(p1_path, dpi=300)
    print('Plot saved to ' + p1_path)
    plt.close()
    alpha_t = np.load(os.path.join(data_dir, 'alpha_t.npy'))
    plt.figure(figsize=(8, 6))
    plt.plot(time_lag[1:], alpha_t[1:], 'b-', linewidth=2)
    plt.axhline(1.0, color='k', linestyle=':', label='alpha=1 (Brownian)')
    plt.axhline(2.0, color='k', linestyle='-.', label='alpha=2 (Ballistic)')
    plt.xlabel('Lag Time [simulation units]')
    plt.ylabel('Local Power-Law Exponent alpha(t)')
    plt.title('Time-Dependent Diffusion Exponent')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    p2_path = os.path.join(data_dir, 'alpha_t_plot_2_' + str(timestamp) + '.png')
    plt.savefig(p2_path, dpi=300)
    print('Plot saved to ' + p2_path)
    plt.close()
    pdfs = np.load(os.path.join(data_dir, 'pdfs.npy'), allow_pickle=True)
    bin_centers = np.load(os.path.join(data_dir, 'bin_centers.npy'), allow_pickle=True)
    tail_exponents = np.load(os.path.join(data_dir, 'tail_exponents.npy'))
    lag_times_pdf = np.load(os.path.join(data_dir, 'lag_times_pdf.npy'))
    plt.figure(figsize=(10, 8))
    colors = ['C0', 'C1', 'C2', 'C3']
    for i in range(len(lag_times_pdf)):
        bc = bin_centers[i]
        pdf = pdfs[i]
        slope = tail_exponents[i]
        plt.loglog(bc, pdf, marker='o', linestyle='', color=colors[i], label='Lag time = ' + str(round(lag_times_pdf[i], 2)))
        if not np.isnan(slope):
            valid_idx = np.where(pdf > 0)[0]
            if len(valid_idx) > 0:
                x_valid = bc[valid_idx]
                y_valid = pdf[valid_idx]
                threshold = np.percentile(x_valid, 80)
                tail_mask = x_valid > threshold
                if np.sum(tail_mask) > 1:
                    x_tail = x_valid[tail_mask]
                    y_tail = y_valid[tail_mask]
                    intercept = np.mean(np.log(y_tail) - slope * np.log(x_tail))
                    x_fit = np.linspace(x_tail[0], x_tail[-1], 10)
                    y_fit = np.exp(intercept) * x_fit**slope
                    plt.loglog(x_fit, y_fit, color=colors[i], linestyle='-', linewidth=2)
    plt.xlabel('Displacement [domain units]')
    plt.ylabel('Probability Density')
    plt.title('Displacement Probability Density Functions')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    p3_path = os.path.join(data_dir, 'PDFs_plot_3_' + str(timestamp) + '.png')
    plt.savefig(p3_path, dpi=300)
    print('Plot saved to ' + p3_path)
    plt.close()
    vacf_norm = np.load(os.path.join(data_dir, 'vacf_norm.npy'))
    t_vacf = time_lag[:len(vacf_norm)]
    plt.figure(figsize=(8, 6))
    plt.plot(t_vacf, vacf_norm, 'g-', linewidth=2)
    plt.axhline(0.0, color='k', linestyle='--')
    plt.xlabel('Lag Time [simulation units]')
    plt.ylabel('Normalized Velocity Autocorrelation Function')
    plt.title('Velocity Autocorrelation Function (VACF)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    p4_path = os.path.join(data_dir, 'VACF_plot_4_' + str(timestamp) + '.png')
    plt.savefig(p4_path, dpi=300)
    print('Plot saved to ' + p4_path)
    plt.close()
    FTLE = np.load(os.path.join(data_dir, 'FTLE.npy'))
    residence_time = np.load(os.path.join(data_dir, 'residence_time.npy'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(residence_time, FTLE, alpha=0.5, color='purple')
    ax1.set_xlabel('Residence Time in Q > 0 Regions [simulation units]')
    ax1.set_ylabel('Finite-Time Lyapunov Exponent (FTLE)')
    ax1.set_title('FTLE vs. Vortex Residence Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.hist(FTLE, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Finite-Time Lyapunov Exponent (FTLE)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of FTLE')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    p5_path = os.path.join(data_dir, 'FTLE_plot_5_' + str(timestamp) + '.png')
    plt.savefig(p5_path, dpi=300)
    print('Plot saved to ' + p5_path)
    plt.close()
    Q_slice = np.load(os.path.join(data_dir, 'Q_slice_z0.npy'))
    Q_slice = np.squeeze(Q_slice)
    final_pos = np.load(os.path.join(data_dir, 'final_wrapped_pos_500.npy'))
    plt.figure(figsize=(8, 6))
    vmin, vmax = np.nanpercentile(Q_slice, [5, 95])
    im = plt.imshow(Q_slice.T, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', cmap='RdBu_r', alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Q-criterion')
    mask = np.abs(final_pos[:, 2]) < 0.1
    if np.sum(mask) > 0:
        plt.scatter(final_pos[mask, 0], final_pos[mask, 1], color='black', s=15, label='Tracers (|z| < 0.1)')
    else:
        plt.scatter([], [], color='black', s=15, label='Tracers (|z| < 0.1)')
    plt.xlabel('x [domain units]')
    plt.ylabel('y [domain units]')
    plt.title('Q-criterion Field (z=0) and Tracer Positions')
    plt.legend(loc='upper right')
    plt.tight_layout()
    p6_path = os.path.join(data_dir, 'Q_slice_plot_6_' + str(timestamp) + '.png')
    plt.savefig(p6_path, dpi=300)
    print('Plot saved to ' + p6_path)
    plt.close()
    print('\nAll plots generated successfully.')