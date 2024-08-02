"""
Simulates Brillouin Pulse Compression.

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
import time
from LASER import LASER
from brillouin_pusher import *




def main( vers ):
    """
    Runs the SBS compression simulation and outputs 2D arrays of pump, stokes,
    and acoustic fields evolution. 
    Plots initial & final profiles; final acoustic field; photon count over time;
    max amplitude variation over time; stokes FWHM over time.
    Normalized by phonon lifetime.
    Can choose between Plasmanik and Drogomir version by arg 'plasm' or 'drogo'.
    """
    
    # initialize LASER object with all simulation parameters
    # l = LASER(xmax=200,tmax=50,
    #              xgrid=0.05,tgrid=0.1 ,
    #              sigp=3,sigs=3,
    #              xp=50,xs=120,
    #              apump=0.2,astokes=0.1,
    #              n=1,gb=0.9,omega=3,G=1,
    #              profile=1)
    
    # l = LASER(xmax=40,tmax=20,
    #               xgrid=0.1,tgrid=0.1,
    #               sigp=10,sigs=1,
    #               xp=10,xs=30,
    #               apump=.2,astokes=.2,
    #               n=1.33,omega=7.42,G=0.539,
    #               profile=2)
    
    # iavor
    # l = LASER(xmax=20,tmax=10,
    #               xgrid=0.01,tgrid=0.01,
    #               sigp=1.6,sigs=0.06,
    #               xp=7,xs=15,
    #               apump=0.095,astokes=0.095,
    #               n=1.33,omega=86.5,G=2*np.pi,
    #               profile=2)
    
    # # \tau_p = 0.3ns, a=0.0217
    # l = LASER(xmax=100,tmax=10,
    #               xgrid=0.08,tgrid=0.08,
    #               sigp=10,sigs=0.4,
    #               xp=30,xs=65,
    #               apump=31,astokes=31,
    #               n=1.3,omega=6.22,G=2*np.pi,
    #               profile=2)
    
    # # # \tau_p = 0.4ns, a=0.01625
    # l = LASER(xmax=100,tmax=40,
    #               xgrid=0.07,tgrid=0.07,
    #               sigp=7.51,sigs=0.3,
    #               xp=30,xs=65,
    #               apump=4.397,astokes=4.397,
    #               n=1.3,omega=8.29,G=2*np.pi,
    #               profile=2)
    
    # # # \tau_p = 0.5ns, a=0.013
    # l = LASER(xmax=100,tmax=40,
    #               xgrid=0.06,tgrid=0.06,
    #               sigp=6,sigs=0.24,
    #               xp=30,xs=65,
    #               apump=4.92,astokes=4.92,
    #               n=1.3,omega=10.367,G=2*np.pi,
    #               profile=2)
    

    # l = LASER(xmax=100,tmax=20,
    #               xgrid=0.1,tgrid=0.1,
    #               sigp=15,sigs=0.6,
    #               xp=30,xs=65,
    #               apump=31,astokes=31,
    #               n=1.3,omega=2.073,G=2*np.pi,
    #               profile=2)
    
    # \tau_p = 1
    # l = LASER(xmax=40,tmax=20,
    #               xgrid=0.05,tgrid=0.05,
    #               sigp=2,sigs=1,
    #               xp=15,xs=21,
    #               apump=.2,astokes=.1,
    #               n=1.29,omega=3,G=1,
    #               profile=1)
    
    # \tau_p = 10
    # l = LASER(xmax=400,tmax=200,
    #               xgrid=0.5,tgrid=0.5,
    #               sigp=20,sigs=10,
    #               xp=150,xs=210,
    #               apump=.2,astokes=.1,
    #               n=1.3,omega=.3,G=.1,
    #               profile=1)
    
    #fiber 
    # l = LASER(xmax=40,tmax=5,
    #               xgrid=0.05,tgrid=0.05,
    #               sigp=10,sigs=3,
    #               xp=15,xs=35,
    #               apump=1,astokes=.1,
    #               n=3.5,omega=804.248,G=2*np.pi,
    #               profile=3)
    
    l = LASER(xmax=30,tmax=4,
                  xgrid=0.01,tgrid=0.01,
                  sigp=4.26321,sigs=0.200187,
                  xp=10,xs=22,
                  apump=1.25138,astokes=1.25138,
                  n=1.3,omega=12.4407,G=2*np.pi,
                  profile=2)
    

    # run simulation
    if vers == 'drogo':
        pump, stokes, rho = brillouin_push(l, a=3.61111)
    elif vers == 'plasm':
        pump, stokes, rho = plasmanik_push(l, a=1)

    # plot results
    plt.style.use(['default'])
    plt.figure()
    plt.plot(l.x, np.abs( stokes[0,:] ), 'r--', label = r'$A_s(z,t)$')
    plt.plot(l.x, np.abs( pump[0,:] ), 'k--', label=r'$A_p(z,0)$')
    plt.plot(l.x, np.abs( stokes[-2,:] ), 'r', label = r'$A_s(z,$' + str(round(l.tmax//1)) + ')')
    plt.plot(l.x, np.abs( pump[-2,:] ), 'k',label= r'$A_p(z,$' + str(round(l.tmax//1)) + ')')
    plt.xlabel('z'); plt.ylabel('A(z,t)')
    plt.legend()
    #plt.gca().set_aspect(l.xmax/l.apump)
    # plt.rc('font', **font)
    plt.savefig('res'+str(l.xgrid//1)+'_'+str(l.tgrid//1)+'_'+str(l.omega//1)+'_'+str(l.G//1)+'.svg')
    
    # acoustic field
    plt.figure()
    plt.plot(l.x, np.abs( rho[-2,:] ) )
    plt.ylabel('$ |p(z,t)|$')
    plt.xlabel('z')
    plt.title('t = '+str( l.tmax ) )
    plt.savefig('aco'+str(l.xgrid//1)+'_'+str(l.tgrid//1)+'_'+str(l.omega//1)+'_'+str(l.G//1)+'.svg')
    
    
    #check energy conservation
    photons = energy_conserv(l, pump, stokes)
    fig, ax = plt.subplots()
    ax.plot(l.t, photons / photons[0], 'k')
    ax.ticklabel_format(useOffset=False)
    plt.ylabel('$\int dz (|A_1|^2 + |A_2|^2)$')
    plt.xlabel(r'$t/\tau_p$')
    plt.savefig('en'+str(l.xgrid//1)+'_'+str(l.tgrid//1)+'_'+str(l.omega//1)+'_'+str(l.G//1)+'.svg')
    
     
    # check amplitude variation  
    # ------ CHANGE h PARAMETER in anlz_peaks if weird results
    amp1, amp2 = anlz_peaks(l, pump, stokes, 0.02)
    plt.figure(5)
    plt.plot(l.t, amp1, label = 'Pump')
    plt.plot(l.t, amp2, label = 'Stokes')
    plt.xlabel(r'$t/\tau_p$')
    plt.ylabel('$max[A]$')
    plt.legend()
    plt.savefig('amp'+str(l.xgrid//1)+'_'+str(l.tgrid//1)+'_'+str(l.omega//1)+'_'+str(l.G//1)+'.svg')
    
    
    # check pulse compression
    plt.figure()
    plt.plot(l.t, FWHM(l, pump, stokes))
    plt.ylabel('FWHM')
    plt.xlabel(r'$t/\tau_p$')
    plt.legend()
    print('max compression : ' + str( np.min( FWHM(l, pump, stokes) ) ) )
    plt.savefig('comp'+str(l.xgrid//1)+'_'+str(l.tgrid//1)+'_'+str(l.omega//1)+'_'+str(l.G//1)+'.svg')
    
        
    return l, pump, stokes, rho





# run simulation
start_time = time.time()

l, pump, stokes, rho = main( 'drogo' )

print("--- %s seconds ---" % (time.time() - start_time))


# %%


# # compare different amplitudes
# plt.figure()
# fig1, ax1 = plt.subplots()

# plt.figure()
# fig2, ax2 = plt.subplots()

# for a in [0.2, 0.3, 0.5]:
#     print(a)

#     l = LASER(xmax=200,tmax=50,
#                  xgrid=0.1,tgrid=0.1,
#                  sigp=10,sigs=5,
#                  xp=50,xs=120,
#                  apump=a,astokes=0.1,
#                  n=1,omega=3,G=1,
#                  profile=1)
    

#     # run simulation
#     pump, stokes, rho = brillouin_push(l, 0.5)

#     # plot results
#     plt.figure()
#     plt.plot(l.x, np.abs( stokes[0,:] ), 'r--', label = 'stokes: t=0')
#     plt.plot(l.x, np.abs( pump[0,:] ), 'k--', label='pump: t=0')
#     plt.plot(l.x, np.abs( stokes[-2,:] ), 'r', label = 'stokes: t=' + str(l.tmax//1))
#     plt.title('SSFM: ' + '$a_{pump} = $' + str(a) )
#     plt.plot(l.x, np.abs( pump[-2,:] ), 'k',label='pump: t=' + str(l.tmax//1))
#     plt.xlabel(r'$z/c\tau_p$'); plt.ylabel('A(z,t)')
#     plt.legend()
#     plt.show()

#     # check energy conservation  
#     photons = energy_conserv(l, pump, stokes)
#     plt.figure(fig1)
#     ax1.plot(l.t, photons / photons[0], label = '$a_p = $'+str(a))
#     ax1.ticklabel_format(useOffset=False)
#     plt.ylabel('$\int dz (|A_1|^2 + |A_2|^2)$')
#     plt.xlabel(r'$t/\tau_p$')
#     plt.legend()
#     plt.show()
    
#     # check pulse compression
#     plt.figure(fig2)
#     ax2.plot(l.t, FWHM(l, pump, stokes), label = '$a_p = $'+str(a))
#     plt.ylabel('FWHM')
#     plt.xlabel(r'$t/\tau_p$')
#     plt.legend()
#     plt.show()
    
#     # plot amplitude variation
#     amp1, amp2 = anlz_peaks(l, pump, stokes, 0.02)
#     plt.figure()
#     plt.plot(l.t, amp1, label = 'pump')
#     plt.plot(l.t, amp2, label = 'stokes')
#     plt.xlabel(r'$t/\tau_p$')
#     plt.ylabel('$max[A]$')
#     plt.legend()





