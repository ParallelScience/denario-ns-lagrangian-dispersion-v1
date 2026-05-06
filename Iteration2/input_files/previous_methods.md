1. **Data Acquisition and Pre-processing**:
   - Download 200 evenly spaced VTK snapshots (indices 18903–19902) using a robust manager.
   - Implement a streaming loader that maintains only two consecutive snapshots in memory to facilitate linear time-interpolation during sub-stepping.
   - Compute the vorticity field $\omega = \nabla \times v$ and the Q-criterion field $Q = \frac{1}{2}(||\Omega||^2 - ||S||^2)$ for each snapshot using central finite differences with periodic boundary wrapping.

2. **Lagrangian Tracer Integration**:
   - Initialize 5,000 tracers at random positions within the domain.
   - Use an RK4 integration scheme with 10 sub-steps per snapshot interval.
   - Employ `scipy.interpolate.RegularGridInterpolator` (linear method) for velocity field interpolation, ensuring periodic boundary conditions are enforced via modulo arithmetic at each sub-step.
   - Store full trajectories as NumPy arrays for subsequent analysis.

3. **Conditional MSD Analysis**:
   - Define "Trapped" and "Free" cohorts based on the distribution of total residence time in high-Q regions (top 20th percentile for "Trapped", bottom 20th percentile for "Free").
   - Compute the MSD separately for each cohort to quantify the retardation effect of vortex structures on the diffusion coefficient.
   - Compare these results to the ensemble-averaged MSD to isolate the impact of coherent structures on local transport.

4. **Vortex Trapping Dynamics**:
   - Calculate the autocorrelation function of the Q-criterion along individual tracer trajectories to determine the characteristic decorrelation time of trapping events.
   - Compare this decorrelation time with the global vortex lifetime to test the hypothesis that 3D vortex instability truncates trapping events.

5. **Finite-Size Saturation Assessment**:
   - Partition the tracer population into "Boundary-Crossers" and "Non-Crossers" based on whether their trajectory crossed the periodic domain boundary.
   - Compute the MSD for both subsets to quantify the influence of the periodic box on late-time $\alpha(t)$ oscillations and verify if "Non-Crossers" maintain a more stable diffusive regime.

6. **Anisotropy and Driving Scale Analysis**:
   - Define the "parallel" direction relative to the local large-scale velocity field (filtered at $n=1-3$).
   - Compute $MSD_{\parallel}$ and $MSD_{\perp}$ and the ratio $\lambda = MSD_{\parallel} / MSD_{\perp}$ to identify preferential transport pathways.
   - Analyze the evolution of $\lambda(t)$ to determine if observed anisotropy is a transient effect of the forcing realization or a persistent feature, and estimate the Lagrangian integral time scale $\tau_c$.

7. **Local Chaotic Dynamics (FTLE)**:
   - Calculate the FTLE along the trajectories of a representative subset (1,000 tracers) by computing the deformation gradient tensor of a small neighborhood around each tracer.
   - Correlate these local stretching rates with the conditional MSD results to quantify how chaotic vortex interactions influence tracer dispersion.

8. **Data Aggregation and Visualization**:
   - Generate log-log plots of MSD(t) for the conditional cohorts and the full ensemble to extract power-law exponents $\alpha(t)$.
   - Create spatial visualizations of tracer density overlaid with Q-criterion isosurfaces to illustrate the interaction between tracers and coherent structures.
   - Produce summary plots of $\alpha(t)$ and $\lambda(t)$ to document the evolution of the diffusion regime and anisotropy.