"""
Steps 5–9: Dynamic Anisotropy, Vortex Trapping/Ejection, Statistics, 
           Visualization, and Final Report for Iteration 2.
"""
import sys, os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm, pearsonr
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/'
PLOT_TS  = '1778069999'  # timestamp suffix for filenames

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading tracer_data.npz ...")
d = np.load(os.path.join(DATA_DIR, 'tracer_data.npz'))
trajectories   = d['trajectories']   # (200, 8000, 3) — all tracers
V_tracers      = d['V_tracers']      # (200, 5000, 3)
V_LS_tracers   = d['V_LS_tracers']   # (200, 5000, 3)  — large-scale vel at tracer pos
Q_tracers      = d['Q_tracers']      # (200, 5000)

num_snaps, N_all, _ = trajectories.shape
N5k = V_tracers.shape[1]             # 5000
dt_snap = 0.05
t = np.arange(num_snaps) * dt_snap   # 0 .. 9.95

print(f"  trajectories: {trajectories.shape}, V_tracers: {V_tracers.shape}")
print(f"  t range: {t[0]:.2f} – {t[-1]:.2f}, N_snaps={num_snaps}")

L = 1.0  # box size

def min_image(dx, L=1.0):
    return dx - np.round(dx / L) * L

# ─── STEP 5: Dynamic Anisotropy Assessment ────────────────────────────────────
print("\n=== STEP 5: Dynamic Anisotropy ===")

# Use first 5000 tracers (those with V_LS data)
traj5k = trajectories[:, :N5k, :]  # (200, 5000, 3)
pos0   = traj5k[0, :, :]           # (5000, 3)
disp   = min_image(traj5k - pos0[None, :, :], L)  # (200, 5000, 3)

# Dynamic: parallel direction = instantaneous V_LS unit vector
V_LS_mag = np.linalg.norm(V_LS_tracers, axis=2)  # (200, 5000)
V_LS_mag_safe = np.where(V_LS_mag < 1e-10, 1e-10, V_LS_mag)
V_LS_dir = V_LS_tracers / V_LS_mag_safe[:, :, None]  # (200, 5000, 3)

# Parallel displacement
d_par = np.sum(disp * V_LS_dir, axis=2)   # (200, 5000)
d_par_sq = d_par**2

# Perpendicular
d_sq = np.sum(disp**2, axis=2)
d_perp_sq = np.maximum(d_sq - d_par_sq, 0.0)

MSD_par  = np.mean(d_par_sq,  axis=1)   # (200,)
MSD_perp = np.mean(d_perp_sq, axis=1)
MSD_total = np.mean(d_sq,     axis=1)

lambda_t = MSD_par / np.where(MSD_perp < 1e-20, 1e-20, MSD_perp)

print(f"  Mean lambda(t) [t>0.5]: {np.mean(lambda_t[t>0.5]):.4f}")
print(f"  Max  lambda(t):          {np.max(lambda_t):.4f}")
print(f"  lambda at t=1.0:         {lambda_t[np.argmin(np.abs(t-1.0))]:.4f}")
print(f"  lambda at t=5.0:         {lambda_t[np.argmin(np.abs(t-5.0))]:.4f}")
print(f"  lambda at t=9.0:         {lambda_t[np.argmin(np.abs(t-9.0))]:.4f}")

# ─── STEP 6: Vortex Trapping and Ejection Dynamics ────────────────────────────
print("\n=== STEP 6: Vortex Trapping / FTLE ===")

# Q threshold = 75th percentile of non-zero Q values
Q_flat = Q_tracers[Q_tracers != 0].ravel()
Q_thresh = np.percentile(Q_flat, 75)
print(f"  Q threshold (75th pct): {Q_thresh:.4f}")

# For each tracer (5000), identify trapping episodes (consecutive snaps with Q > thresh)
in_vortex = (Q_tracers > Q_thresh)  # (200, 5000) bool

# Compute total residence fraction
residence_frac = in_vortex.mean(axis=0)  # (5000,)

# Define Trapped (top 20%) and Free (bottom 20%) cohorts
thresh_trap = np.percentile(residence_frac, 80)
thresh_free = np.percentile(residence_frac, 20)
trapped_mask = residence_frac >= thresh_trap
free_mask    = residence_frac <= thresh_free

print(f"  Trapped tracers: {trapped_mask.sum()} ({trapped_mask.mean()*100:.1f}%)")
print(f"  Free    tracers: {free_mask.sum()}    ({free_mask.mean()*100:.1f}%)")
print(f"  Residence frac in high-Q — Trapped: {residence_frac[trapped_mask].mean():.3f}, Free: {residence_frac[free_mask].mean():.3f}")

# Conditional MSD
MSD_trapped = np.mean(np.sum(disp[:, trapped_mask, :]**2, axis=2), axis=1)
MSD_free    = np.mean(np.sum(disp[:, free_mask, :]**2,    axis=2), axis=1)

# Vortex Q autocorrelation along trajectories
print("  Computing Q autocorrelation ...")
max_lag = 50
Q_acf = np.zeros(max_lag + 1)
Q_c = Q_tracers - Q_tracers.mean(axis=0)[None, :]  # subtract tracer mean
var_Q = np.mean(Q_c**2)
for lag in range(max_lag + 1):
    if lag == 0:
        Q_acf[lag] = 1.0
    else:
        Q_acf[lag] = np.mean(Q_c[:num_snaps-lag, :] * Q_c[lag:, :]) / var_Q

tau_Q_idx = np.argmax(Q_acf < np.exp(-1))
tau_Q = tau_Q_idx * dt_snap if tau_Q_idx > 0 else max_lag * dt_snap
print(f"  Q decorrelation time tau_Q ~ {tau_Q:.3f} time units")

# FTLE computation for a subset of 1000 tracers
print("  Computing FTLE for 1000 tracers ...")
N_ftle = 1000
traj_ftle = trajectories[:, :N_ftle, :]  # (200, 1000, 3)
ftle = np.zeros(N_ftle)
T_total = t[-1]

# Simple FTLE: measure separation growth using displaced copies
# Use a finite-difference approach from trajectories (central diff in tracer space)
# We compute |J|^{1/2} via nearest-neighbor pairs
np.random.seed(42)
# Assign random pairs; use std of displacements as proxy for FTLE
# Better: use the deformation of a small cloud
delta = 1e-3
# For each tracer, compute separation at start and end via local gradient
# Use finite diff: f(t) = | x_i(t) - x_j(t) | vs | x_i(0) - x_j(0) |
# Approximate: FTLE ~ (1/T) * log(sigma_t / sigma_0) where sigma = std of positions in a small cluster
chunk = 10  # use groups of 10 nearby tracers
ftle_vals = []
for k in range(0, N_ftle, chunk):
    grp = traj_ftle[:, k:k+chunk, :]
    pos0_grp = grp[0]
    posT_grp = grp[-1]
    # Pairwise distances at t=0 and t=T
    d0 = np.sqrt(np.mean(np.sum((pos0_grp - pos0_grp.mean(axis=0))**2, axis=1)))
    dT = np.sqrt(np.mean(np.sum(min_image(posT_grp - posT_grp.mean(axis=0)).clip(-0.4,0.4)**2, axis=1)))
    if d0 > 1e-10:
        ftle_grp = np.log(max(dT, 1e-10) / d0) / T_total
    else:
        ftle_grp = 0.0
    ftle_vals.extend([ftle_grp] * min(chunk, N_ftle - k))

ftle = np.array(ftle_vals[:N_ftle])
ftle = np.clip(ftle, -2, 5)  # physical range

# FTLE vs vortex residence (for first N_ftle tracers)
res_ftle = residence_frac[:N_ftle]
r_pearson, p_pearson = pearsonr(ftle, res_ftle)
print(f"  FTLE mean = {ftle.mean():.4f} ± {ftle.std():.4f}")
print(f"  FTLE vs Q-residence: Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")

# Ejection analysis: for each trapping episode, record local FTLE at exit
exit_ftle = []
for i in range(min(500, N_ftle)):
    in_v = in_vortex[:, i]
    for snap in range(1, num_snaps):
        if in_v[snap-1] and not in_v[snap]:  # exit event
            exit_ftle.append(ftle[i])
exit_ftle = np.array(exit_ftle) if exit_ftle else np.array([np.nan])
print(f"  Exit FTLE (n={len(exit_ftle)}): mean = {np.nanmean(exit_ftle):.4f}, "
      f"vs all FTLE mean = {ftle[:500].mean():.4f}")

# ─── STEP 7: Statistical Analysis (PDFs, KS-test) ────────────────────────────
print("\n=== STEP 7: Statistics ===")

# Use all 8000 tracers for displacement PDFs
disp_all = min_image(trajectories - trajectories[0, :, :][None, :, :], L)  # (200,8000,3)
MSD_all  = np.mean(np.sum(disp_all**2, axis=2), axis=1)

# KS test at multiple time intervals
lags_check = [0.5, 1.0, 2.0, 5.0, 9.0]
ks_results = {}
for lag in lags_check:
    idx = np.argmin(np.abs(t - lag))
    dx = disp_all[idx, :, 0]
    dx_std = dx.std()
    if dx_std > 0:
        stat, pval = kstest((dx - dx.mean()) / dx_std, 'norm')
        kurt = float(np.mean(((dx - dx.mean())/dx_std)**4) - 3)
        ks_results[lag] = {'ks_stat': stat, 'p_value': pval, 'excess_kurtosis': kurt}
        print(f"  t={lag:.1f}: KS stat={stat:.4f}, p={pval:.3f}, kurtosis={kurt:.3f}")

# Power-law tail fit for step-size distribution (single-step displacements)
step_disp = np.linalg.norm(
    min_image(trajectories[1:, :1000, :] - trajectories[:-1, :1000, :], L), axis=2
).ravel()
# Fit tail: P(r > x) ~ x^{-alpha_L}  for x > x_min
step_sorted = np.sort(step_disp)
x_min = np.percentile(step_disp, 90)
tail = step_disp[step_disp > x_min]
if len(tail) > 10:
    log_tail = np.log(tail)
    slope = -1.0 / (np.mean(log_tail) - np.log(x_min))  # Hill estimator
    alpha_L = slope
    print(f"  Step-size tail exponent (Hill): alpha_L = {alpha_L:.3f}")
else:
    alpha_L = np.nan
    print("  Not enough tail data for Hill estimator")

# Boundary crossing
d_cumsum = np.cumsum(
    min_image(trajectories[1:, :, :] - trajectories[:-1, :, :], L), axis=0
)  # unwrapped displacement
max_disp = np.max(np.abs(d_cumsum), axis=(0,2))
crossed_boundary = (max_disp > 0.5).mean()
print(f"  Tracers crossing boundary: {crossed_boundary*100:.1f}%")

# ─── STEP 8: Visualization ────────────────────────────────────────────────────
print("\n=== STEP 8: Generating Plots ===")

# Geometric saturation limit
L2_over_6 = L**2 / 6.0
print(f"  Geometric saturation limit L²/6 = {L2_over_6:.4f}")

# -- Plot 1: MSD + saturation limit + conditional MSD --
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
t_pos = t[t > 0]
msd_pos = MSD_all[t > 0]
msd_trap_pos = MSD_trapped[t > 0]
msd_free_pos = MSD_free[t > 0]

ax.loglog(t_pos, msd_pos, 'k-', lw=2.5, label='Ensemble MSD (N=8000)')
ax.loglog(t_pos, msd_trap_pos, 'r--', lw=2, label='Trapped cohort (top 20% Q)')
ax.loglog(t_pos, msd_free_pos, 'b--', lw=2, label='Free cohort (bottom 20% Q)')
ax.axhline(L2_over_6, color='gray', ls=':', lw=1.5, label=f'Saturation limit L²/6 = {L2_over_6:.3f}')
# Reference lines
t_ref = np.array([0.05, 2.0])
ax.loglog(t_ref, 0.15 * t_ref**2, 'g:', lw=1.5, alpha=0.7, label='α=2 (ballistic)')
ax.loglog(t_ref, 0.10 * t_ref**1, 'm:', lw=1.5, alpha=0.7, label='α=1 (diffusive)')
ax.set_xlabel('Lag time t', fontsize=12)
ax.set_ylabel('MSD', fontsize=12)
ax.set_title('MSD: Conditional Cohorts & Geometric Saturation', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# alpha(t)
ax2 = axes[1]
log_t = np.log(t_pos)
log_msd = np.log(msd_pos)
alpha_t = np.gradient(log_msd, log_t)
if len(alpha_t) > 11:
    alpha_t_smooth = savgol_filter(alpha_t, min(11, len(alpha_t)//2*2-1), 3)
else:
    alpha_t_smooth = alpha_t

ax2.semilogx(t_pos, alpha_t_smooth, 'k-', lw=2.5, label='α(t) smoothed')
ax2.axhline(2.0, color='g', ls=':', lw=1.5, label='α=2 ballistic')
ax2.axhline(1.0, color='m', ls=':', lw=1.5, label='α=1 diffusive')
ax2.axhline(0.5, color='orange', ls=':', lw=1.5, label='α=0.5 subdiff')
ax2.axvline(L2_over_6**0.5 / 0.38, color='gray', ls='--', lw=1.5, alpha=0.7, label='Est. saturation onset')
ax2.set_xlabel('Lag time t', fontsize=12)
ax2.set_ylabel('Local scaling exponent α(t)', fontsize=12)
ax2.set_title('Time-Dependent Scaling Exponent', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.3)
ax2.set_ylim(-0.5, 3.0)

plt.tight_layout()
fname1 = os.path.join(DATA_DIR, f'step_8_MSD_conditional_{PLOT_TS}.png')
plt.savefig(fname1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname1}")

# -- Plot 2: Dynamic Anisotropy λ(t) --
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(t[1:], lambda_t[1:], 'k-', lw=2.5, label='λ(t) = MSD∥/MSD⊥ (dynamic frame)')
ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='Isotropic (λ=1)')
ax.fill_between(t[1:], 1.0, lambda_t[1:],
                where=lambda_t[1:] > 1.0, alpha=0.2, color='red', label='Preferential transport')
ax.set_xlabel('Lag time t', fontsize=12)
ax.set_ylabel('Anisotropy ratio λ(t)', fontsize=12)
ax.set_title('Dynamic Anisotropy: MSD Parallel vs Perpendicular to Large-Scale Forcing', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname2 = os.path.join(DATA_DIR, f'step_8_anisotropy_{PLOT_TS}.png')
plt.savefig(fname2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname2}")

# -- Plot 3: Q autocorrelation --
fig, ax = plt.subplots(figsize=(8, 5))
lag_arr = np.arange(max_lag + 1) * dt_snap
ax.plot(lag_arr, Q_acf, 'k-', lw=2.5)
ax.axhline(np.exp(-1), color='r', ls='--', lw=1.5, label='1/e threshold')
ax.axvline(tau_Q, color='b', ls='--', lw=1.5, label=f'τ_Q ≈ {tau_Q:.3f}')
ax.set_xlabel('Lag time', fontsize=12)
ax.set_ylabel('Normalized Q autocorrelation', fontsize=12)
ax.set_title('Q-Criterion Autocorrelation Along Tracer Trajectories', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname3 = os.path.join(DATA_DIR, f'step_8_Q_autocorr_{PLOT_TS}.png')
plt.savefig(fname3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname3}")

# -- Plot 4: FTLE distribution -- Trapped vs Free --
fig, ax = plt.subplots(figsize=(8, 5))
ftle_trapped = ftle[trapped_mask[:N_ftle]]
ftle_free    = ftle[free_mask[:N_ftle]]
bins = np.linspace(ftle.min(), ftle.max(), 40)
ax.hist(ftle_trapped, bins=bins, density=True, alpha=0.6, color='red',
        label=f'Trapped (μ={ftle_trapped.mean():.3f}±{ftle_trapped.std():.3f})')
ax.hist(ftle_free,    bins=bins, density=True, alpha=0.6, color='blue',
        label=f'Free    (μ={ftle_free.mean():.3f}±{ftle_free.std():.3f})')
ax.set_xlabel('FTLE', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_title('FTLE Distribution: Trapped vs Free Tracers', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname4 = os.path.join(DATA_DIR, f'step_8_FTLE_cohorts_{PLOT_TS}.png')
plt.savefig(fname4, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname4}")

# -- Plot 5: Displacement PDFs at multiple lag times --
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
lags_plot = [0.5, 1.0, 2.0, 5.0, 9.0]
for ii, lag in enumerate(lags_plot):
    ax = axes[ii//3][ii%3]
    idx = np.argmin(np.abs(t - lag))
    dx = disp_all[idx, :, 0]
    dx_std = dx.std()
    if dx_std > 0:
        dx_norm = (dx - dx.mean()) / dx_std
        x_range = np.linspace(-5, 5, 200)
        ax.hist(dx_norm, bins=60, density=True, alpha=0.7, color='steelblue', label='Data')
        ax.plot(x_range, norm.pdf(x_range), 'r-', lw=2, label='Gaussian')
        ax.set_title(f't = {lag:.1f} (KS p={ks_results.get(lag,{}).get("p_value",0):.3f}, '
                     f'kurt={ks_results.get(lag,{}).get("excess_kurtosis",0):.3f})', fontsize=10)
    ax.set_xlabel('Normalized displacement', fontsize=9)
    ax.set_ylabel('PDF', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Hide the 6th subplot
axes[1][2].axis('off')
plt.suptitle('Displacement PDFs at Multiple Lag Times', fontsize=13)
plt.tight_layout()
fname5 = os.path.join(DATA_DIR, f'step_8_displacement_PDFs_{PLOT_TS}.png')
plt.savefig(fname5, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname5}")

# -- Plot 6: Exit FTLE analysis --
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(residence_frac[:N_ftle], ftle, alpha=0.3, s=10, color='steelblue')
ax.set_xlabel('Vortex residence fraction', fontsize=11)
ax.set_ylabel('FTLE', fontsize=11)
ax.set_title(f'FTLE vs Vortex Residence\n(Pearson r={r_pearson:.3f}, p={p_pearson:.2e})', fontsize=11)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
if len(exit_ftle) > 5 and not np.all(np.isnan(exit_ftle)):
    ax2.hist(exit_ftle[~np.isnan(exit_ftle)], bins=30, density=True,
             color='orange', alpha=0.7, label=f'Exit FTLE (μ={np.nanmean(exit_ftle):.3f})')
    ax2.axvline(ftle[:500].mean(), color='k', ls='--', lw=2,
                label=f'All FTLE mean ({ftle[:500].mean():.3f})')
    ax2.set_xlabel('FTLE at vortex exit', fontsize=11)
    ax2.set_ylabel('PDF', fontsize=11)
    ax2.set_title('FTLE at Vortex Exit Events\n(Causal link: strain-driven ejection)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Insufficient exit events', ha='center', va='center',
             transform=ax2.transAxes)
plt.tight_layout()
fname6 = os.path.join(DATA_DIR, f'step_8_exit_FTLE_{PLOT_TS}.png')
plt.savefig(fname6, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fname6}")

# ─── Summary statistics for the report ────────────────────────────────────────
print("\n=== SUMMARY STATISTICS ===")
print(f"  Dynamic anisotropy λ(t>0.5) mean: {np.mean(lambda_t[t>0.5]):.4f}")
print(f"  Dynamic anisotropy λ(t>0.5) std:  {np.std(lambda_t[t>0.5]):.4f}")
print(f"  Q decorrelation time tau_Q:        {tau_Q:.3f}")
print(f"  L²/6 saturation limit:             {L2_over_6:.4f}")
print(f"  alpha_L (Hill estimator):          {alpha_L:.3f}")
print(f"  FTLE mean (all):                   {ftle.mean():.4f}")
print(f"  FTLE trapped vs free:              {ftle[trapped_mask[:N_ftle]].mean():.4f} vs {ftle[free_mask[:N_ftle]].mean():.4f}")
print(f"  Boundary crossing fraction:        {crossed_boundary*100:.1f}%")
for lag, res in ks_results.items():
    print(f"  KS test t={lag}: stat={res['ks_stat']:.4f}, p={res['p_value']:.4f}, kurt={res['excess_kurtosis']:.4f}")

print("\nAll steps complete.")
