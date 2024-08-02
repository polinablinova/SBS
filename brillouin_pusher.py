"""
Implements Split Step Fourier Method to simulate laser pulse compression in
stimulated Brillouin scattering process.

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

from LASER import LASER


        
        
     
def brillouin_push(l: LASER, a):
    """
    Simulates SBS process for given simulation parameters in LASER object
    ----------
    l : LASER object
    a : coupling constant
    """

    nt, nx = len(l.t), len(l.x)
   
    A1, A2 = l.pump, l.stokes

    # stores time evolution
    A1_store = np.zeros((nt, nx), dtype = 'complex')
    A2_store = np.zeros((nt, nx), dtype = 'complex')
    q_store = np.zeros((nt, nx), dtype = 'complex')
    
    A1_store[0, :] = A1
    A2_store[0, :] = A2
    
    
    # aliasing filter 
    k, w = fftfreq( len(l.x), l.xgrid), fftfreq( len( l.t ), l.tgrid )
    gw = fft( gain(l) ) 
    gw = gdealiase(l, w, gw)
    g = ifft( gw )
    #g = gain(l)
    
    
    for i in range(1,nt):
        
        # keep fixed boundaries
        A1[0] = 0; A1[-1] = 0; A2[0] = 0; A2[-1] = 0;
        
        
        # act with propagation operator
        k, kA1, kA2 = adialiase( l, k, fft(A1), fft(A2) )
        
        A1 = ifft( np.exp( -(1j) * 2 * np.pi * k * l.tgrid / l.n) * kA1  )
        A2 = ifft( np.exp( (1j) * 2 * np.pi * k * l.tgrid / l.n) * kA2  )
        
        
        # update history
        A1_store[i,:], A2_store[i,:] = A1, A2


        # q convolutions for each position for the current time step
        qs = np.zeros((nx), dtype = 'complex')
        
        for j in range(1,nx-1):
            
            # convolution integral (eq. 8c)
            q = ifft(  fft( g ) * 
                     fft( A1_store[:,j] * np.conjugate( A2_store[:,j]) )  ) /np.sqrt(2 * np.pi) 
            
            qs[j] = q[i]
            
        
        # apply nonlinearity
        A1, A2, A1_store, A2_store = nonlinearity(l,A1,A2,A1_store,A2_store,qs,i,a)
        q_store[i,:] = qs 
        
        
    return A1_store, A2_store, q_store



def plasmanik_push(l: LASER, a):
    """
    Simulates SBS process with acoustic wave not dependent on time for given 
    simulation parameters in LASER object 
    ----------
    l : LASER object
    a : coupling constant
    """

    nt, nx = len(l.t), len(l.x)
   
    A1, A2 = l.pump, l.stokes

    # stores time evolution
    A1_store = np.zeros((nt, nx), dtype = 'complex')
    A2_store = np.zeros((nt, nx), dtype = 'complex')
    q_store = np.zeros((nt, nx), dtype = 'complex')
    qs = np.zeros((nx), dtype = 'complex')
    
    
    A1_store[0, :] = A1
    A2_store[0, :] = A2
    
    
    # aliasing filter 
    k, w = fftfreq( len(l.x), l.xgrid), fftfreq( len( l.t ), l.tgrid )
    
    for i in range(1,nt):
        
        # keep fixed boundaries
        A1[0] = 0; A1[-1] = 0; A2[0] = 0; A2[-1] = 0;
        
        
        # act with propagation operator
        k, kA1, kA2 = adialiase( l, k, fft(A1), fft(A2) )
        
        A1 = ifft( np.exp( -(1j) * 2 * np.pi * k * l.tgrid/l.n ) * kA1  )
        A2 = ifft( np.exp( (1j) * 2 * np.pi * k * l.tgrid/l.n ) * kA2  )
        
        qs[:] = - np.sqrt(2 * np.pi) * A1 * np.conjugate( A2 ) * (1j) / l.G
        
        # update history
        A1_store[i,:], A2_store[i,:] = A1, A2
            
        
        # apply nonlinearity
        A1, A2, A1_store, A2_store = nonlinearity(l,A1,A2,A1_store,A2_store,qs,i,a)
        q_store[i,:] = qs 
        
        
    return A1_store, A2_store, q_store
        



# nonlinearity operator
@numba.jit(nopython=True) 
def nonlinearity(l:LASER, A1, A2, A1_store, A2_store, qs, i, a):
    """
    Applies nonlinearity 
    ----------
    l : LASER object
    A1, A2 : 1D pump/stokes arrays holding x values at current time step
    A1_store, A2_store : 2D array storing history of pump/stokes in [t,x] format
    qs : 1D array holding convolution results for each z at current time step
    i : current time step
    a : coupling constant
    """
    
    # edges are fixed at 0 so need to avoid division by zero error
    aq = np.abs(qs[1:-1])
    cq = np.conjugate(qs[1:-1])
    qs = qs[1:-1]
    a1, a2 = A1[1:-1], A2[1:-1]
    
    A1[1:-1] = np.cos( a * l.tgrid/l.n * aq) * a1 - (1j) * qs * np.sin(a * l.tgrid/l.n * aq) * a2 / aq
    A2[1:-1] = np.cos( a * l.tgrid/l.n * aq) * a2 - (1j) * cq * np.sin(a * l.tgrid/l.n * aq) * a1 / aq
    
    A1_store[i,:] = A1
    A2_store[i,:] = A2
    
    return A1, A2, A1_store, A2_store






@numba.jit(nopython=True) 
def gain(las:LASER):
    """
    Gain function
    ----------
    l : LASER object
    """
    
    nt = len(las.t)
    dt = las.tgrid
    G = las.G
    Om = las.omega
    
    gain_array = np.zeros(nt, 'complex')
    
    for i in range(nt): 
        if i >= 0:
            gain_array[i] = (-Om  *  np.sqrt( 2 * np.pi )  *  np.exp( - G * i*dt / 2 )  
                             *  np.exp( (1j) * Om * i*dt)  *  
                             np.sin( np.sqrt( Om**2 - G**2 / 4 ) * i*dt )  /  
                             np.sqrt( Om**2 - G**2 / 4 ) )
        
        else:
            gain_array[i] = 0
            
    return gain_array   





@numba.jit(nopython=True)    
def gdealiase(l: LASER, w, gw):
    """
    Applies two-thirds dealiasing filter to gain function
    """          
    wmax = np.max(w)
    for wi in range( len(w) ) :
        if np.abs( w[wi] ) > 2*wmax/3 :
            gw[ wi ] = 0
            
           
    return gw
        


@numba.jit(nopython=True)  
def adialiase(l:LASER, k, A1k, A2k):
    """
    Applies two-thirds dealiasing filter A1 and A2
    """  
    kmax = 2* np.pi * np.max(k)
    
    # set high-frequency components to zero
    k[ k >  2*kmax/3 ] = 0 
    

    for ki in range( len(k) ) :
        if np.abs( k[ki] ) > 2*kmax/3 :
            A1k[ ki ], A2k[ ki ] = 0, 0

            
    return k, A1k, A2k






# ----------- MIGHT NEED TO CHANGE HEIGHT PARAMETER 
def anlz_peaks(l:LASER, pump, stokes, h):
    """
    Gives peak amplitude values for each time step
    ----------
    l : LASER object holding simulation parameters
    pump : 2D array with pump pulse evolution
    stokes : 2D array with stokes pulse evolution
    h: minimum height that counts for a peak
    """
    AMPS1 = np.empty( (len(l.t)) )
    AMPS2 = np.empty( (len(l.t)) )
    
    for i in range(len(l.t)):
        a1, a2 = pump[i,:], stokes[i,:]
        
        peaks, d = find_peaks(np.abs(a1), height = h)
        AMPS1[i] = max(d['peak_heights'])
        
        peaks2, d2 = find_peaks(np.abs(a2), height = h)
        AMPS2[i] = max(d2['peak_heights'])
        
    return AMPS1, AMPS2





def energy_conserv(l:LASER, pump, stokes):
    """
    Computes photon number at each time step
    ----------
    l : LASER object
    pump : 2D array storing pump evolution 
    stokes : 2D array storing stokes evolution
    """
    energy = np.empty( (len(l.t)) )
    
    for i in range(len(l.t)):
        a1, a2 = pump[i,:], stokes[i,:]
        
        E = np.real( a1 * np.conjugate(a1) + a2 * np.conjugate(a2) )
        E[np.isnan(E)] = 0
        energy[i] = np.sum(E)
        
    return energy




@numba.jit(nopython=True)
def FWHM(l:LASER, pump, stokes):
    """
    Computes FWHM pulse duration of the Stokes pulse
    ----------
    l : LASER object
    pump : 2D array storing pump evolution 
    stokes : 2D array storing stokes evolution
    """
    widths = np.zeros( len(l.t) )
    for tind in range(len(stokes[:,0])):
        
        st = stokes[tind,:]
        st[np.isnan(st)] = 0
        fwhm = np.nonzero(np.absolute(st) > np.absolute(np.max(np.absolute(st))/2.0))
        fwhm = (fwhm[0][-1] - fwhm[0][0]) * l.xgrid
        widths[tind] = fwhm
        
    return widths



