# denario-ns-lagrangian-dispersion-v1

**Scientist:** denario-6
**Date:** 2026-05-05

## Dataset: 3D Driven Turbulence Simulation — Lagrangian Tracer Dispersion

### Source
HuggingFace dataset: https://huggingface.co/datasets/pedrota2000/NS_simulation
Downloaded snapshots are stored locally in: /home/node/work/projects/ns_lagrangian_v1/data/

### Simulation Overview
3D isothermal hydrodynamic driven turbulence simulation run with AthenaK (https://github.com/PrincetonUniversity/athenak). The simulation models subsonic turbulence driven on large scales in a periodic box. Turbulence is continuously forced at large scales (wavenumber n=1-3, peak at n=2) with purely solenoidal driving (sol_fraction=1.0), producing a statistically stationary turbulent velocity field.

### Simulation Parameters
- Code: AthenaK
- Problem: 3D driven turbulence
- Grid resolution: 128 × 128 × 128 cells
- Domain: [-0.5, 0.5]³ (unit box with periodic boundary conditions on all faces)
- EOS: Isothermal, sound speed c_s = 5.0
- Time integrator: RK2, CFL = 0.3
- Spatial reconstruction: PLM, Riemann solver: HLLE
- Turbulence driving: dedt = 1e-4, tcorr = 5.0, nlow=1, nhigh=3
- Mach number: subsonic, rms Mach ~ 0.07 (rms velocity ~ 0.38, sound speed 5.0)

### File Inventory
- Format: Legacy VTK binary (DATASET STRUCTURED_POINTS)
- Naming convention: Turb.hydro_w.NNNNN.vtk where NNNNN is the output index
- Total snapshots: 1001 files, output indices 18903–19903 (last file 19903 appears truncated — use indices 18903–19902 only, i.e., 1000 usable snapshots)
- Time range: t ≈ 189.03 to 199.02 (output cadence Δt = 0.01, total span ≈ 10 time units)
- Grid size: 128³ = 2,097,152 cells, cell size Δx = 0.0078125

Pre-downloaded sample snapshots (for quick testing):
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.18903.vtk  — t=189.03 (start)
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.18950.vtk  — t=189.50
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19000.vtk  — t=190.00
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19100.vtk  — t=191.00
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19200.vtk  — t=192.00
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19400.vtk  — t=194.00
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19600.vtk  — t=196.00
- /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.19800.vtk  — t=198.00

Additional snapshots must be downloaded on demand from HuggingFace:
  URL template: https://huggingface.co/datasets/pedrota2000/NS_simulation/resolve/main/Turb.hydro_w.NNNNN.vtk
  Example: wget -q "https://huggingface.co/datasets/pedrota2000/NS_simulation/resolve/main/Turb.hydro_w.18910.vtk" -O /home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.18910.vtk

### Reading a VTK File (Python)
```python
import pyvista as pv
import numpy as np

mesh = pv.read("/home/node/work/projects/ns_lagrangian_v1/data/Turb.hydro_w.18903.vtk")
# Reshape to 3D grid (cell-centred, 128³)
dens = mesh['dens'].reshape(128, 128, 128)   # float32, mass density ρ ≈ 1.0 ± 0.01
velx = mesh['velx'].reshape(128, 128, 128)   # float32, velocity v_x in [-0.87, 0.69]
vely = mesh['vely'].reshape(128, 128, 128)   # float32, velocity v_y in [-0.69, 0.72]
velz = mesh['velz'].reshape(128, 128, 128)   # float32, velocity v_z in [-0.79, 0.85]
s00  = mesh['s_00'].reshape(128, 128, 128)   # float32, passive scalar/tracer field ≈ 0.047

# Grid coordinates (cell centres)
# x_i = -0.5 + (i + 0.5) * 0.0078125  for i in 0..127
# Same for y and z
```

### Data Fields (cell-centred)
| Field | Description | Dtype | Approx range |
|-------|-------------|-------|-------------|
| dens  | Mass density ρ | float32 | [0.985, 1.007] |
| velx  | Velocity v_x | float32 | [-0.87, 0.69] |
| vely  | Velocity v_y | float32 | [-0.69, 0.72] |
| velz  | Velocity v_z | float32 | [-0.79, 0.85] |
| s_00  | Passive scalar (tracer) | float32 | ≈ 0.047 (uniform) |

### Physical Units and Scale
- Domain: unit box [-0.5, 0.5]³, cell size Δx = 1/128 ≈ 0.0078125
- Time unit: simulation time units, cadence Δt = 0.01, total span ~10 time units
- Velocity: rms |v| ≈ 0.38, max |v| ≈ 1.04, sound speed c_s = 5.0 → subsonic (Mach ~ 0.07)
- Density: nearly uniform (isothermal), σ(ρ)/⟨ρ⟩ ≈ 0.001
- Turbulence correlation time: t_corr = 5.0 (driving), so the 10-unit time span covers ~2 correlation times
- Eddy turnover time (estimated): L/v_rms ≈ 1.0/0.38 ≈ 2.6 time units

### Research Tasks

The primary research questions are:

1. **Lagrangian tracer dispersion:** Initialize many passive tracers (N_tracers ≥ 1000, e.g. 2000–5000) randomly in the domain at t=t_0. Evolve each tracer's position by integrating dx/dt = v(x(t), t) using trilinear interpolation of the velocity field on the 128³ grid. Apply periodic boundary conditions (domain [-0.5, 0.5]). Use RK2 or RK4 time integration with sub-stepping to resolve velocity gradients.

2. **MSD and RMSD:** Compute the ensemble-averaged mean-square displacement MSD(t) = ⟨|r(t) - r(0)|²⟩ and RMSD(t) = √MSD(t) as a function of lag time. Account for periodic boundaries when computing displacements (use the minimum image convention). Also compute the MSD for each spatial component separately: MSD_x, MSD_y, MSD_z.

3. **Scaling law:** Fit MSD(t) ~ t^α using log-log regression (power law). Classify the diffusion regime:
   - α ≈ 1: normal diffusion (Brownian)
   - α = 2: ballistic transport
   - 1 < α < 2: superdiffusion
   - 0 < α < 1: subdiffusion
   Identify any crossover in scaling at different time scales (e.g. ballistic at early times, diffusive at late times, or persistent superdiffusion).

4. **Anomalous diffusion and Lévy flights:** Analyze whether anomalous diffusion is present. Fit the distribution of individual tracer displacements to heavy-tailed distributions (power laws, Lévy stable distributions). Compute the velocity autocorrelation function (VACF) and its time integral (Taylor's diffusion formula). Identify coherent trapping/ejection events by analyzing tracer trajectories near vortex structures. Extract vorticity field ω = ∇ × v and identify vortex cores (Q-criterion or λ₂-criterion). Measure waiting-time and step-size distributions of tracers to test the Lévy flights hypothesis. Compute the Lévy stability index α_L from the step-size PDF tails.

5. **Chaotic vortex interactions as the physical mechanism:** Quantify the degree of chaos in vortex interactions using finite-time Lyapunov exponents (FTLE) from the tracer trajectories. Investigate whether tracers spend anomalously long times trapped in vortex cores (sticky regions in phase space) before being ejected — this is the physical mechanism for Lévy flights in 2D/3D turbulence. Compute the vorticity PDF, Q-criterion field, and correlate vortex presence with anomalous displacement events.

### Suggested Analysis Strategy
- For manageability: use every 10th or every 5th snapshot (100–200 time steps) from the full 1001 available — download as needed. For quick prototyping, use the 8 pre-downloaded snapshots; for the final analysis, download ~50–200 evenly spaced files.
- Use scipy.interpolate.RegularGridInterpolator (or equivalent) for trilinear interpolation.
- Use parallel processing (up to 8-16 workers) for tracer integration.
- Save tracer trajectories as numpy arrays for downstream analysis.
- All figures should use descriptive titles, axis labels, and log-log scales for power-law fitting.

### Notes and Caveats
- The simulation is 3D, isothermal, subsonic turbulence — NOT 2D. Lévy flights in 3D turbulence are less common than in 2D, but coherent structures still produce anomalous transport.
- The time span (~10 units, ~2-4 eddy turnover times) may not be long enough to observe the full diffusive limit — be honest about this.
- Periodic boundaries must be respected in both tracer advection and displacement calculation.
- The last VTK file (19903) appears truncated — exclude it from the analysis.
- Download only the files you need — each is ~40 MB. The full dataset is ~40 GB; do not attempt to download all 1001 files.
