1. **Data Acquisition and Management**:
   - Implement a robust download manager to fetch 200 evenly spaced VTK snapshots from the HuggingFace repository. Include error handling (try-except blocks) and file integrity checks (size verification) to manage potential network interruptions.
   - Implement a streaming data loader that keeps only two consecutive snapshots ($t_n$ and $t_{n+1}$) in memory at any time to prevent memory overflow.
   - Pre-compute the vorticity field $\omega = \nabla \times v$ using central finite differences with periodic boundary wrapping to ensure accuracy at grid edges.

2. **Lagrangian Tracer Initialization**:
   - Initialize 10,000 passive tracers with uniform random coordinates $(x, y, z)$ within the domain $[-0.5, 0.5]^3$.
   - Store initial positions $r(t_0)$ as the reference for displacement calculations.

3. **Tracer Advection and Integration**:
   - Implement an RK4 integration scheme to evolve tracer positions.
   - Perform integration between snapshots $t_n$ and $t_{n+1}$ using linear interpolation in time for the velocity field: $v(r, t) = v(r, t_n) \cdot (1 - \tau) + v(r, t_{n+1}) \cdot \tau$, where $\tau = (t - t_n) / \Delta t$.
   - Use `scipy.interpolate.RegularGridInterpolator` for trilinear spatial interpolation.
   - Apply periodic boundary conditions at each sub-step using the modulo operator: $r_{new} = (r_{old} + v \cdot dt) \pmod{1.0} - 0.5$.
   - Use at least 5–10 sub-steps per snapshot interval to ensure numerical stability.

4. **Displacement and MSD Computation**:
   - Calculate the displacement vector $\Delta r(t) = r(t) - r(0)$ using the minimum image convention: $\Delta r_i = \Delta r_i - \text{round}(\Delta r_i / L) \cdot L$, where $L=1.0$.
   - Compute the ensemble-averaged MSD and its components ($MSD_x, MSD_y, MSD_z$).
   - Quantify anisotropy by comparing MSD components; if no global mean flow exists, focus on local anisotropy relative to the local vorticity vector.

5. **Scaling Law and Diffusion Regime Analysis**:
   - Perform log-log linear regression on $MSD(t)$ vs $t$ to estimate the power-law exponent $\alpha$.
   - Calculate the time-dependent exponent $\alpha(t)$ using a Savitzky-Golay filter on the log-log data to obtain a stable derivative and identify transitions between ballistic and diffusive regimes.

6. **Anomalous Diffusion and Statistical Analysis**:
   - Compute the probability density function (PDF) of tracer displacements at multiple time intervals.
   - Fit the tails of the displacement PDFs to Lévy stable distributions to extract the stability index $\alpha_L$.
   - Calculate the Velocity Autocorrelation Function (VACF) and its integral to compare with MSD results via Taylor’s diffusion formula.

7. **Coherent Structure and Chaos Analysis**:
   - Identify vortex cores using the Q-criterion ($Q > 0$).
   - Compute Finite-Time Lyapunov Exponents (FTLE) for a subset of 500 tracers by integrating neighboring particles to compute the deformation gradient tensor.
   - Correlate tracer residence times within high-Q regions with anomalous displacement events to test the "sticky vortex" hypothesis.

8. **Data Aggregation and Visualization**:
   - Aggregate trajectory data into structured NumPy arrays.
   - Generate log-log plots of MSD(t) with standard error bars.
   - Create spatial maps of tracer density overlaid with Q-criterion isosurfaces to visualize the interaction between tracers and turbulent structures.