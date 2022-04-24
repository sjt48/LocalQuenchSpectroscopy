"""
Exact Diagonalisation code to illustrate local quench spectroscopy.

---------------------------------------------
S. J. Thomson
Dahlem Centre for Complex Quantum Systems, FU Berlin
steven.thomson@fu-berlin.de
steventhomson.co.uk / @PhysicsSteve
https://orcid.org/0000-0001-9065-9842
---------------------------------------------

This work is licensed under a Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License. This work may
be edited and shared by others, provided the author of this work is credited 
and any future authors also freely share and make their work available. This work
may not be used or modified by authors who do not make any derivative work 
available under the same conditions. This work may not be modified and used for
any commercial purpose, in whole or in part. For further information, see the 
license details at https://creativecommons.org/licenses/by-nc-sa/4.0/.

This work is distributed without any form of warranty or committment to provide technical 
support, nor the guarantee that it is free from bugs or technical incompatibilities
with all possible computer systems. Use, modify and troubleshoot at your own risk.

If you do use any of this code, please cite https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.033337.

"""
#==============================================================================
# Initialisation
#==============================================================================

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from scipy import fftpack
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from quspin.tools.block_tools import block_ops
from multiprocessing import freeze_support
from datetime import datetime
startTime = datetime.now()      # Initialise timer

# Adds LaTeX labels, sets figure size
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8,6)
plt.rc('font',family='serif')
plt.rcParams.update({'font.size': 24})
plt.rc('text', usetex=True)
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

#==============================================================================
# Define variables
#==============================================================================
L = 11                          # System size 
                                # Use odd numbers so the flip is on the central spin
#------------------------------------------------------------------------------
# Hamiltonian Parameters
#------------------------------------------------------------------------------
J0 = 1.                         # Kinetic term (\sigma^x_i * \sigma^x_{i+1})
h = 3.                          # On-site term (\sigma_z)
                                
#==============================================================================
# Exact Diagonalisation code
#==============================================================================
def ED(h,J0):
    # Create site-coupling lists for periodic boundary conditions
    J = [[-J0,i%L,(i+1)%L] for i in range(L)]
    hlist = [[-h,i] for i in range(L)]
        
    # Create static and dynamic lists for constructing Hamiltonian
    # (The dynamic list is zero because the Hamiltonian is time-independent)
    static = [["xx",J],["z",hlist]]
    dynamic = []
    
    # Create Hamiltonian as block_ops object
    # This splits the Hamiltonian into different momentum subspaces which can be diagonalised independently
    blocks=[dict(kblock=kblock) for kblock in range(L)]     # Define the momentum blocks
    basis_args = (L,)                                       # Mandatory basis arguments
    basis_kwargs = dict(S='1/2',pauli='1')
    H_block = block_ops(blocks,static,dynamic,spin_basis_1d,basis_args,basis_kwargs=basis_kwargs,dtype=np.complex128,check_symm=False)
    
    # Set up list of observables to calculate
    # (Density on each site.)
    basis = spin_basis_1d(L,S='1/2',pauli='1')                    # Create 1D spin basis
    no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
    n_list = [hamiltonian([["y",[[1.0,i]]]],[],basis=basis,dtype=np.complex128,**no_checks) for i in range(L)]
    
    # Define initial state as superposition of two product states in the Hilbert space
    # (This is equivalent to the spin rotation, but it's easier to do it like this in the ED code.)
    st = "".join("1" for i in range(L//2)) + "0" + "".join("1" for i in range(L//2)) 
    iDH = basis.index(st)
    st_0 = "".join("1" for i in range(L))
    i0 = basis.index(st_0)
    psi1 = np.zeros(basis.Ns)
    psi1[i0] = np.sqrt(0.5)
    psi1[iDH] = np.sqrt(0.5)

    # Time evolution of state
    times = np.linspace(0.0,20,150,endpoint=True)
    psi1_t = H_block.expm(psi1,a=-1j,start=times[0],stop=times[-1],num=len(times),block_diag=True)
    
    # Time evolution of observables
    n_t = np.vstack([n.expt_value(psi1_t).real for n in n_list]).T
    
    steps = len(times)
    dt = times[1] - times[0]
    # Compute Fourier transform of observables, and lists of momenta/energies
    ft = np.abs(np.fft.fft2(n_t, norm=None))
    ft = np.fft.fftshift(ft)
    energies = fftpack.fftshift(fftpack.fftfreq(steps) * (2.*np.pi/dt))
    # Normalise FT
    data = np.abs(ft)/np.max(np.abs(ft))

    # Plot density dynamics
    fig=plt.imshow(n_t[::-1],aspect = 'auto',interpolation=None,extent=(1,L,times[0],times[-2]))
    plt.colorbar(fig, orientation = 'vertical',extend='max')
    plt.xlabel(r'$R$')
    plt.ylabel(r'$t$')
    plt.text(11.5,20,r'$\langle n_R (t) \rangle$')
    plt.savefig('fig1.png',dpi=150,bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Compute the exact result for the spectrum for plotting purposes
    spectrum = np.zeros(L)
    krange = np.linspace(-np.pi,np.pi,L)
    for i in range(L):
        spectrum[i] = 2*np.sqrt(h**2+J0**2-2*h*J0*np.cos(krange[i]))

    # Plot 2D Fourier transform
    plt.imshow(data[::-1], aspect = 'auto', interpolation = 'none' , extent = (-np.pi, np.pi, max(energies)/J0, min(energies)/J0))
    plt.plot(krange,spectrum,'r--')
    plt.plot(krange,-1*spectrum,'r--')
    plt.xlabel(r'$k/a$')
    plt.ylabel(r'$E/J_0$')
    plt.savefig('fig2.png',dpi=150,bbox_inches='tight')
    plt.show()
    plt.close()
        
    print('End',datetime.now()-startTime)

#------------------------------------------------------------------------------
if __name__ == '__main__':

    freeze_support()
    ED(h,J0)
