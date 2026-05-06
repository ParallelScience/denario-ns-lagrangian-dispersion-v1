1. **Data Acquisition and Pre-processing**:
   - Download 200 evenly spaced VTK snapshots (indices 18903–19902).
   - Implement a streaming loader that maintains only two consecutive snapshots in memory, using explicit memory management (e.g., `del`, `gc.collect()`) to prevent leaks.
   - For each snapshot, compute the vorticity field $\omega = \nabla \times v$ and the Q-criterion field $Q = \frac{1}{2}(||\Omega||^2 - ||S||^2)$ using central finite differences with periodic boundary wrapping.
   - Perform a 3D FFT on the velocity grid to extract the large-scale solenoidal driving component ($n=1-3$) by zeroing out high-wavenumber modes and performing an inverse FFT. Cache these filtered fields for interpolation.

2. **Lagrangian Tracer Integration**:
   - Initialize 5,000 tracers at random positions.
   - Use an RK4 integration scheme with 10 sub-steps per snapshot interval.
   - Employ `scipy.interpolate.RegularGridInterpolator` (linear method) for both the full velocity field and the pre-filtered large-scale velocity field $\mathbf{V}_{LS}(\mathbf{x}, t)$.
   - Enforce periodic boundary conditions via modulo arithmetic at each sub-step.
   - Use parallel processing to distribute the integration of tracers across CPU cores.

3. **Refined Scaling and Saturation Analysis**:
   - Compute the ensemble-averaged MSD(t) and component-wise $MSD_x, MSD_y, MSD_z$ using the minimum image convention.
   - Plot the theoretical geometric saturation limit ($L^2/6$) on log-log plots.
   - Calculate the Lagrangian Integral Time Scale ($\tau_L$) via the Velocity Autocorrelation Function (VACF) to validate the independence of early-time transport from box size.
   - Attribute late-time deviations in the power-law exponent $\alpha(t)$ to geometric saturation.

4. **Dynamic Anisotropy Assessment**:
   - Define the parallel and perpendicular displacement components relative to the interpolated instantaneous local large-scale velocity vector $\mathbf{V}_{LS}(\mathbf{x}, t)$.
   - Calculate the dynamic anisotropy ratio $\lambda(t) = MSD_{\parallel}(t) / MSD_{\perp}(t)$ to determine if anisotropy persists or decays as tracers lose memory of their initial state.

5. **Vortex Trapping and Ejection Dynamics**:
   - Define "Trapped" cohorts as tracers residing in high-Q regions ($Q > Q_{threshold}$).
   - Track entry and exit times to identify full trapping events.
   - Correlate residence time with local FTLE values at the point of exit to test if ejection is driven by high-strain regions.

6. **Local Chaotic Dynamics (FTLE)**:
   - Calculate the FTLE for a subset of 1,000 tracers to manage computational load.
   - Compute the flow map gradient $\nabla \Phi$ using a small perturbation ($\delta \approx 0.01 \Delta x$) to track the deformation of a local particle cluster (Jacobian approach).
   - Quantify the transition from vortex-dominated transport (low FTLE) to strain-dominated transport (high FTLE).

7. **Statistical Rigor and Distribution Analysis**:
   - Compute displacement PDFs at multiple time intervals (e.g., $0.5\tau_e, 1.0\tau_e, 2.0\tau_e$).
   - Use Maximum Likelihood Estimation (MLE) to fit power-law tails and estimate the Lévy stability index $\alpha_L$.
   - Perform a Kolmogorov-Smirnov test against a Gaussian distribution to quantify the degree of non-Gaussianity and distinguish between true Lévy flights and sweeping-induced heavy tails.

8. **Data Aggregation and Visualization**:
   - Generate summary plots of $\alpha(t)$ and $\lambda(t)$.
   - Create spatial visualizations of tracer density overlaid with Q-criterion isosurfaces and FTLE ridges.
   - Produce a comparative plot of residence time vs. local FTLE to demonstrate the causal link between vortex breakdown and transport.