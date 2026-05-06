# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib.pyplot as plt
import datetime

plt.rcParams['text.usetex'] = False

def generate_plots():
    data_dir = 'data/'
    print('Loading data...')
    step3_data = np.load(os.path.join(data_dir, 'step_3_results.npz'))
    t = step3_data['t']
    msd_ensemble = step3_data['msd_ensemble']
    msd_trapped = step3_data['msd_trapped']
    msd_free = step3_data['msd_free']
    msd_crossers = step3_data['msd_crossers']
    msd_non_crossers = step3_data['msd_non_crossers']
    alpha_t = step3_data['alpha_t']
    lambda_t = step3_data['lambda_t']
    vacf = step3_data['vacf']
    taylor_diff = step3_data['taylor_diff']
    step4_data = np.load(os.path.join(data_dir, 'step_4_results.npz'))
    R_tau = step4_data['R_tau']
    FTLE = step4_data['FTLE']
    trapped_idx = step4_data['trapped_idx_1000']
    free_idx = step4_data['free_idx_1000']
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
    axs1[0, 0].loglog(t[1:], msd_ensemble[1:], label='Ensemble MSD', color='black', lw=2)
    axs1[0, 0].set_xlabel('Time t')
    axs1[0, 0].set_ylabel('MSD')
    axs1[0, 0].set_title('Ensemble Mean-Square Displacement')
    axs1[0, 0].grid(True, which='both', ls='--', alpha=0.7)
    axs1[0, 0].legend()
    axs1[0, 1].loglog(t[1:], msd_trapped[1:], label='Trapped', color='blue', lw=2)
    axs1[0, 1].loglog(t[1:], msd_free[1:], label='Free', color='orange', lw=2)
    axs1[0, 1].set_xlabel('Time t')
    axs1[0, 1].set_ylabel('MSD')
    axs1[0, 1].set_title('Conditional MSD (Trapped vs Free)')
    axs1[0, 1].grid(True, which='both', ls='--', alpha=0.7)
    axs1[0, 1].legend()
    axs1[1, 0].plot(t[1:], alpha_t, color='purple', lw=2)
    axs1[1, 0].axhline(1.0, color='k', linestyle='--', label='Normal Diffusion (alpha=1)')
    axs1[1, 0].axhline(2.0, color='r', linestyle='--', label='Ballistic (alpha=2)')
    axs1[1, 0].set_xlabel('Time t')
    axs1[1, 0].set_ylabel('alpha(t)')
    axs1[1, 0].set_title('Power-law Exponent alpha(t)')
    axs1[1, 0].grid(True, alpha=0.7)
    axs1[1, 0].legend()
    axs1[1, 1].loglog(t[1:], msd_crossers[1:], label='Crossers', color='red', lw=2)
    axs1[1, 1].loglog(t[1:], msd_non_crossers[1:], label='Non-Crossers', color='green', lw=2)
    axs1[1, 1].set_xlabel('Time t')
    axs1[1, 1].set_ylabel('MSD')
    axs1[1, 1].set_title('Boundary-Crossers vs Non-Crossers MSD')
    axs1[1, 1].grid(True, which='both', ls='--', alpha=0.7)
    axs1[1, 1].legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(data_dir, 'msd_results_' + timestamp + '.png'), dpi=300)
    plt.close(fig1)
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    axs2[0].plot(t[1:], lambda_t[1:], color='brown', lw=2)
    axs2[0].axhline(1.0, color='k', linestyle='--', label='Isotropic (lambda=1)')
    axs2[0].set_xlabel('Time t')
    axs2[0].set_ylabel('lambda(t)')
    axs2[0].set_title('Anisotropy Ratio lambda(t)')
    axs2[0].grid(True, alpha=0.7)
    axs2[0].legend()
    axs2[1].plot(t, vacf, color='teal', lw=2)
    axs2[1].axhline(0.0, color='k', linestyle='--')
    axs2[1].set_xlabel('Time t')
    axs2[1].set_ylabel('VACF')
    axs2[1].set_title('Velocity Autocorrelation Function')
    axs2[1].grid(True, alpha=0.7)
    axs2[2].plot(t, taylor_diff, color='magenta', lw=2)
    axs2[2].set_xlabel('Time t')
    axs2[2].set_ylabel('Taylor Diffusion Coefficient')
    axs2[2].set_title('Taylor Diffusion over Time')
    axs2[2].grid(True, alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(os.path.join(data_dir, 'anisotropy_vacf_' + timestamp + '.png'), dpi=300)
    plt.close(fig2)
    fig3 = plt.figure(figsize=(15, 10))
    ax1 = fig3.add_subplot(2, 2, 1)
    ax2 = fig3.add_subplot(2, 2, 2)
    ax3 = fig3.add_subplot(2, 2, 3)
    ax1.plot(t, R_tau, color='navy', lw=2)
    ax1.set_xlabel('Time lag tau')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Q-criterion Autocorrelation')
    ax1.grid(True, alpha=0.7)
    ftle_final = FTLE[-1]
    ax2.hist(ftle_final, bins=30, edgecolor='k', alpha=0.7, color='skyblue')
    ax2.set_xlabel('FTLE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('FTLE Distribution at Final Time')
    ax2.grid(True, alpha=0.7)
    ftle_trapped = ftle_final[trapped_idx]
    ftle_free = ftle_final[free_idx]
    ax3.bar(['Trapped', 'Free'], [np.mean(ftle_trapped), np.mean(ftle_free)], color=['blue', 'orange'])
    ax3.set_ylabel('Mean FTLE')
    ax3.set_title('Mean FTLE: Trapped vs Free')
    fig3.tight_layout()
    fig3.savefig(os.path.join(data_dir, 'vortex_ftle_' + timestamp + '.png'), dpi=300)
    plt.close(fig3)
    print('Plots generated successfully.')

if __name__ == '__main__':
    generate_plots()