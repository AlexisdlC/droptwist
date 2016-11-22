# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:37:34 2016

@author: Alexis
"""

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
from jones_matrix import JonesMatrix as jm
from jones_matrix import Phases as ph
# change the following to %matplotlib notebook for interactive plotting

get_ipython().magic(u'matplotlib inline')

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import numba as nb
import pims
import trackpy as tp
import matplotlib.animation as animation

@nb.njit(nogil=True)
def jones_matrix(gamma, phiex, phiord):
    """ The phase matrix for polarized light
    computed using definition from Liquid Crystal Dispersions by Drzaic.

    In x,y,z coordinate system where light propagates in positive z direction.
    :math:`\\theta` is polar angle, :math:`\\phi` is azimuthal for position.
    :math:`\\rho` is distance from z axis.
    The director is a unit vector with polar angle beta and azimuthal gamma

    Inputs
    ------
    gamma : azimuthal
    phiex : extraordinary phase retardation for this voxel. depends on beta.
    phiord : angle independent ordinary phase retardation

    Returns
    -------
    jones matrix
            complex numpy ndarray with shape (2,2)
    """
    jm = np.array(((0j,0j),(0j,0j)))
    j11 = ((np.cos(gamma)**2)*np.exp(1j*phiex) +
           (np.sin(gamma)**2)*np.exp(1j*phiord))
    j12 = np.cos(gamma)*np.sin(gamma)*(np.exp(1j*phiex) - np.exp(1j*phiord))
    j22 = ((np.cos(gamma)**2)*np.exp(1j*phiord) +
           (np.sin(gamma)**2)*np.exp(1j*phiex))
    #return np.array([[j11, j12], [j12, j22]], dtype=nb.complex128)
    jm[0,0] = j11
    jm[0,1] = j12
    jm[1,0] = j12
    jm[1,1] = j22
    return jm


@nb.njit(nogil=True)
def compute_phiex(beta,
                  ordinary_n,
                  extraordinary_n,
                  wavelen=0.650,
                  resolution=0.011):
    """
    Returns the extraordinary phase shift caused by a field.
    Computations in microns.
    beta is the angle of the director to the optical (z) axis.
    """
    _nExtraBeta_num = ordinary_n*extraordinary_n
    _nExtraBeta = (_nExtraBeta_num /
                   ((np.sqrt(np.sin(beta)*ordinary_n)**2 +
                     (np.cos(beta)*extraordinary_n)**2)))
    return 2*np.pi*_nExtraBeta*resolution/wavelen


# A the ordinary phase component is constant.
@nb.njit(nogil=True)
def compute_phiord(ordinary_n, wavelen=0.650, resolution=0.011):
    """
    Returns the ordinary phase shift caused by a field.
    Computations in microns.
    """
    return 2*np.pi*ordinary_n*resolution/wavelen



    
def f(r,rmin,rmax):
    return 1/np.sqrt(r)*(np.pi*np.log(r/rmin)/np.log(rmax/rmin))


def rd(x,y,z):
    return np.sqrt(x**2+y**2+z**2)

def tet(x,y,z):
    return np.arccos(z/np.sqrt(x**2+y**2+z**2))

def phi(x,y,z):
    return np.arctan(y/x)

def nr(x,y,z,b,rmin,rmax):
    return (1-0.5*(b**2)*(f(rd(x,y,z),rmin,rmax)**2))*rd(x,y,z)

def nphi(x,y,z,b,rmin,rmax):
    return b*f(rd(x,y,z),rmin,rmax)+phi(x,y,z)

def ntet(x,y,z):
    return tet(x,y,z)

def nx(x,y,z,b,rmin,rmax):
    return nr(x,y,z,b,rmin,rmax)*np.sin(ntet(x,y,z))*np.cos(nphi(x,y,z,b,rmin,rmax))

def ny(x,y,z,b,rmin,rmax):
    return nr(x,y,z,b,rmin,rmax)*np.cos(ntet(x,y,z))*np.sin(nphi(x,y,z,b,rmin,rmax))

def nz(x,y,z,b,rmin,rmax):
    return nr(x,y,z,b,rmin,rmax)*np.cos(ntet(x,y,z))

def beta(x,y,z,b,rmin,rmax):
    return np.arccos(nz(x,y,z,b,rmin,rmax))

def gamma(x,y,z,b,rmin,rmax):
    return np.arctan(ny(x,y,z,b,rmin,rmax)/nx(x,y,z,b,rmin,rmax))
    
def polarizer(ang):
    return np.array([[np.cos(ang)**2,np.cos(ang)*np.sin(ang)],
                     [np.cos(ang)*np.sin(ang),np.sin(ang)**2]],
                    dtype=np.complex)


rmax = 20.0
rmin = 0.1
n_ord = 1.55
n_ext = 1.65
lamb = 0.660 #microns
zres = 0.0011 
xres = 0.0075
wlen = lamb/rad


x = np.arange(-1,1,xres)
y = 0.0 # because of symmetry we only need to compute one slice
z = np.arange(-1,1,zres)


def jm_mat(x,y,z,b,rmin,rmax):
    res_jm = np.ones((x.shape[0],2,2))
    #res_jm = np.ones((x.shape[0],2,2),dtype=np.complex)
    res_jm[:,0,1] = 0
    res_jm[:,1,0] = 0
    for i,xi in enumerate(x):
        for zi in z:
            if np.sqrt(xi**2+zi**2) < rmax:
                gam = gamma(xi,y,zi,b,rmin,rmax)
                bet = beta(xi,y,zi,b,rmin,rmax)
                phiex = compute_phiex(bet,n_ord,n_ext,wlen,zres)
                phior = compute_phiord(n_ord,wlen,zres)
                #res_jm[i] = np.dot(res_jm[i],
                #                   jm.jones_matrix(gam,phiex,phior))
                res_jm[i] = np.dot(jones_matrix(gam,phiex,phior),
                                   res_jm[i])
    return res_jm


jms = jm_mat(x,y,z,0,rmin,rmax)


im_row = np.zeros_like(x)
for i in np.arange(x.shape[0]):
    im_row[i] = np.linalg.norm(np.dot(np.dot(polarizer(np.pi/4),jms[i]),
                                      polarizer(-np.pi/4)))

plt.plot(x,im_row)


plt.pcolor(x,np.arange(10),np.tile(im_row,(10,1)),cmap=plt.cm.gray)





