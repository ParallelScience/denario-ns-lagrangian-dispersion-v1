

Iteration 0:
### Summary: Lagrangian Dispersion in 3D Subsonic Isothermal Turbulence

**Dataset & Methodology**
- **Simulation:** 3D isothermal hydrodynamic turbulence (AthenaK), $128^3$ grid, subsonic ($M \approx 0.07$), solenoidal driving ($n=1-3$).
- **Analysis:** 10,000 tracers integrated via RK4 with trilinear interpolation over 10 time units ($\approx 4$ eddy turnover times).
- **Constraints:** Periodic boundaries enforced via minimum image convention; last snapshot (19903) excluded due to truncation.

**Key Findings**
- **Isotropy:** Dispersion is highly isotropic ($\lambda \approx 1.11$); large-scale solenoidal forcing does not induce persistent preferential transport axes.
- **Scaling Regimes:** Confirmed Taylor’s classical dispersion theory. Transition from ballistic ($\alpha \approx 2, t < 0.25$) to diffusive ($\alpha \approx 1, t > 0.8$) regimes.
- **Diffusion Coefficient:** Taylor diffusion coefficient $D \approx 0.0101$, derived from a Lagrangian integral time scale $\tau_c \approx 0.2264$.
- **Absence of Anomalous Diffusion:** Displacement PDFs are Gaussian (excess kurtosis $\approx 0$). Tail analysis ($\alpha_L \approx 8.4–9.1$) rejects Lévy flight hypothesis.
- **Microscopic Dynamics:** "Sticky vortex" hypothesis confirmed locally: tracers in high-Q regions exhibit lower FTLE (mean 0.936) and suppressed separation. However, 3D vortex instability/short lifetimes prevent macroscopic anomalous transport.

**Limitations & Uncertainties**
- **Finite-Size Effects:** 93.5% of tracers crossed periodic boundaries; late-time statistics ($t > 3.5$) are influenced by re-sampling of correlated structures.
- **Time Span:** While sufficient for ballistic-to-diffusive crossover, the simulation duration is insufficient to reach a perfectly stable asymptotic diffusive limit.

**Recommendations for Future Work**
- **Extend Domain/Resolution:** To mitigate periodic boundary re-sampling, future runs should utilize larger domains or longer simulation times if computational resources permit.
- **Vortex Lifetime Analysis:** Quantify the distribution of vortex lifetimes to formally correlate the truncation of "sticky" events with the convergence to Gaussian statistics.
- **Parameter Space Exploration:** Investigate if increasing the Mach number or changing the driving scale ($n$) alters the vortex stability enough to trigger anomalous transport.
        