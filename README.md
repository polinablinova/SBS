# SBS
Numerical simulations of laser pulse compression in gases via stimulated Brillouin scattering (SBS). Three different simulations are implemented and documented below: pseudospectral simulation (SSFM.zip), averaged kinetic simulation (aKin.zip), and a three-wave solution (WAV3.zip)
## Split-Step Fourier Method Solution for Transient SBS
See https://ieeexplore.ieee.org/document/806589 for the model used. The program (SSFM.zip) consists of three files: **LASER.py**, **brillouin_pusher.py**, and **simulator.py**, which implement the simulation parameter class, helper functions, and the simulation function call repectively. The class file and the simulation file are detailed below. simulator.py simply exectutes all the function calls defined in the two other files.
### LASER.py
This file includes the definition for the class LASER, which holds all simulation parameters for SBS run. The main simulation function, and all helper function use this class to reduce the number of used parameters. LASER is compiled with numba’s @jitclass. This class initializes two pump and stokes pulses to be later used in simulation.<br />
**Attributes**<br />
-xmax, tmax: maximum spatial/temporal window size<br />
-xgrid, tgrid: spatial/temporal step<br />
-sigp, sigs: width of pump/stokes pulse (in step units)<br />
-xp, xs: central point of pump/stokes pulse (in step units)<br />
-apump, astokes: amplitude of pump/stokes<br />
-n: refractive index of the Brillouin-active medium<br />
-G: Brillouin linewidth<br />
-omega: Brillouin frequency shift<br />
-x, t: spatial/temporal grid<br />
-pump: array for pump profile. Profile 1: Gaussian; profile 2: flat-top Gaussian<br />
-stokes: array for stokes profile, always Gaussian<br />
All real attributes are stored in single-precision floating-point format (float32), and pump and stokes pulses are stored as double-precision complex numbers (complex128). At the beginning of each simulation, a LASER object is initialized and used throughout the simulation.
### brillouin_pusher.py
This is the main simulation file. It implements the function brillouin push which calculates each step of pump and stokes propagation through the Brillouin-active medium using the split-step-fourier-method (SSFM). The pseudo-code for the SSFM process is as follows:
(1) create 2D-arrays to store evolution of pump and stokes, and store the initial pulse profiles as the first column
(2) Apply aliasing filter to the gain function and the temporal frequencies that will be used throughout all steps
(3) Enter outer time loop
– Fix boundaries at zero
– Apply aliasing filter to pump, stokes, and spatial frequencies
– Compute the effect of propagating operator, and store in history arrays
– Create array for convolution integral calculations
  Enter inner space loop
    ∗ Calculate convolution integrals for each gridpoint as in Eq.(8), and store them in the previously created array
– Apply nonlinearity operator to A1, A2, and store results in history arrays
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
