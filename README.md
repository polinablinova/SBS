# SBS
Numerical simulations of laser pulse compression in gases via stimulated Brillouin scattering (SBS). Three different simulations are implemented and documented below: pseudospectral simulation (SSFM.zip), averaged kinetic simulation (aKin.zip), and a three-wave solution (three_wave.m)
## SSFM.zip: Split-Step Fourier Method Solution for Transient SBS
See https://ieeexplore.ieee.org/document/806589 for the model used. The program (SSFM.zip) consists of three files: **LASER.py**, **brillouin_pusher.py**, and **simulator.py**.
### LASER.py
This file includes the definition for the class LASER, which holds all simulation parameters for SBS run. The main simulation function, and all helper function use this class to reduce the number of used parameters. LASER is compiled with numba’s @jitclass. This class initializes pump and stokes pulses. All real attributes are stored in single-precision floating-point format (float32), and pump and stokes pulses are stored as double-precision complex numbers (complex128). At the beginning of each simulation, a LASER object is initialized and used throughout the simulation.
### brillouin_pusher.py
This is the main simulation file. It implements the function brillouin_push which calculates each step of pump and stokes propagation through the Brillouin-active medium using the split-step-fourier-method (SSFM). The pseudo-code for the SSFM process is as follows:<br />
(1) Create 2D-arrays to store evolution of pump and stokes, and store the initial pulse profiles as the first column<br />
(2) Apply aliasing filter to the gain function and the temporal frequencies that will be used throughout all steps<br />
(3) Enter outer time loop<br />
    – Fix boundaries at zero<br />
    – Apply aliasing filter to pump, stokes, and spatial frequencies<br />
    – Compute the effect of propagating operator, and store in history arrays<br />
    – Create array for convolution integral calculations<br />
      Enter inner space loop<br />
        ∗ Calculate convolution integrals for each gridpoint as in Eq.(8), and store them in the previously created array<br />
    – Apply nonlinearity operator to A1, A2, and store results in history arrays<br />
### simulator.py
Using simulator.py, the laser object can be initialized. For example,
```
l = LASER(xmax=200,tmax=50,
xgrid=0.05,tgrid=0.05,
sigp=10,sigs=5,
xp=50,xs=120,
apump=0.2,astokes=0.1,
n=1,gb=2,omega=3,G=1,
profile=1)
```
and the simulation is ran either with
```
brillouin_push(l, a)
```
or with
```
plasmanik_push(l, a)
```
where the second version assumes that a steady-state in the acoustic field has been reached. a is the gain factor, which is equal to half the product of the Brillouin linewidth and Brillouin gain g.
## aKin.zip: averaged Particle-In-Cell solution
See https://pubs.aip.org/aip/pop/article/11/11/5204/261409/Slowly-varying-envelope-kinetic-simulations-of for the model used. The program (aKin.zip) consists of three files: **damped_pondermotive.m**, **solve_tridiag.m**, and **continuous_pump.m**.
### damped_pondermotive.m
Main simulation file. Contains all simulation parameters and function calls. The laser envelopes are defined on a grid from 0 to xi_max with spacing dxi, which is set equal to the time step dt. The simulation time is set by T_max. Both pump and stokes are defined as Gaussian pulses centered at xp and xs respectively, with amplitudes apump and astokes, and widths sigma_p and sigma_s.<br />
Each laser grid has ppb number of particles (bin), and the particles are seeded uniformly in phase space (P and Phi) in each bin.
The method employs an RK4 solver for particles and solves laser evolution using a tridiagonal solver. The pseudo-code is as follows:<br />
(1) Create 2D-arrays to store evolution of pump and stokes, and store the initial pulse profiles as the first row<br />
(2) Create a 3D-array that stores particle bins (nxi) with a certain number of particles (ppb) for each time step (nt)<br />
(3) Enter the outer time loop<br />
    - Calculate the bounce frequency<br />
      Enter the spatial loop<br />
    * Calculate pondermotive phase (same for each laser gridpoint) and currents<br />
      Enter the particle loop<br />
      ** Solve the equation of motion of each particle using RK4.<br />
    - Propagate lasers<br />
### solve_tridiag.m
A simple tridiagonal solver used as a helper function in dampled_pondermotive.m
### continuous_pump.m
Same as damped_pondermotive but with a continuous pump laser input.
## three_wave.m: simple solution to the fluid three-wave equation
A MATLAB implementation of the three-wave equation (See, e.g., https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.84.1208). A standard Runge-Kutta solver (RK4) is implemented.

