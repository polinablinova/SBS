"""
Holds class LASER for simulating Brillouin Pulse Compression.

Created on Tue Oct 24 17:55:13 2023
Run on a core i7 with python 3.9.12 and win11

@author: Polina Blinova
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, ifft
import numba
from numba import int32, float32, float64, complex128
from numba.experimental import jitclass
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import integrate


# class holding simulation parameters
laserspec = [
    ('xmax', float32),            
    ('tmax', float32),  
    ('xgrid', float32),
    ('tgrid', float32),
    ('pump', complex128[:]),
    ('stokes', complex128[:]),
    ('x', float32[:]),
    ('t', float32[:]),
    ('sigp', float32),
    ('sigs', float32),
    ('xp', float32),
    ('xs', float32),
    ('n', float32),
    ('G', float32),       # Brillouin linewidth
    ('omega', float32),   # Brillouin Shift
    ('apump', float32),
    ('astokes', float32),
    ('profile', int32)
]

@jitclass(laserspec)
class LASER(object):
    def __init__(self, 
                 xmax,tmax,
                 xgrid,tgrid,
                 sigp,sigs,
                 xp,xs,
                 apump,astokes,
                 n,omega,G,profile):
        
        
        self.xmax = xmax        # max window size
        self.tmax = tmax        # max time
        self.xgrid = xgrid      # dx
        self.tgrid = tgrid      # dt
        self.sigp = sigp        # width of pump
        self.sigs = sigs        # width of stokes
        self.xp = xp            # center of pump
        self.xs = xs            # center of stokes
        self.apump = apump      # pump amplitude
        self.astokes = astokes  # stokes amplitude
        self.n = n              # refractive index of Brillouin medium
                                # gain = \alpha * \Gamma_B / 2, NOT USED 
        self.G = G              # Brillouin linewidth
        self.omega = omega      # Brillouin frequency shift
        
        self.x = np.arange(0, xmax, xgrid, dtype = float32)
        self.t = np.arange(0, tmax, tgrid, dtype = float32)
        
        # pump and stokes pulses
        self.pump = np.zeros( (len(self.x)), dtype = complex128 )
        self.stokes = np.zeros( (len(self.x)), dtype = complex128 )
        
        # Gaussian
        if profile == 1:
            self.pump[:] = apump * np.exp(-(self.x - xp) ** 2 / sigp ** 2)
            self.stokes[:] = astokes * np.exp(-(self.x - xs) ** 2 / sigs ** 2)
            
        # Step pump
        if profile == 2:
            self.pump[:] = apump * np.exp(-(self.x - xp) ** 8 / sigp ** 8)
            self.stokes[:] = astokes * np.exp(-(self.x - xs) ** 2 / sigs ** 2)
            
        # Fiber
        tr = 0.01
        Es = 1.41421
        Eb = 0.251487
        if profile == 3:
            self.pump[:] = apump * np.sqrt( ( np.tanh((self.x - xp + sigp/2)/(0.45 * tr)) 
                                           - np.tanh((self.x - xp - sigp/2)/(0.45 * tr)) )/2 )
            self.stokes[:] = Eb + (Es-Eb) * np.sqrt( ( np.tanh((self.x - xs + sigs/2)/(0.45 * tr)) 
                                           - np.tanh((self.x - xs - sigs/2)/(0.45 * tr)) )/2 )
            
            
            
            
            
            