# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:45:45 2015
generalized function for computing a jmimage
analytical form or (interpolated) data
@author: zoeydavidson
"""

import numpy as np
from Phases import *
from JonesMatrix import *
from numba import jit
import inspect

droplet_r = 40.0  # um
droplet_center_x = 50
droplet_center_y = 50
droplet_center_z = 50
ordinary_n = 1.40
extraordinary_n = 1.42

image_size = 100
xy_res = 300
slices = 500
delta_z = 2.0*droplet_r/slices

samplespace_x = np.linspace(0, image_size, xy_res)
samplespace_y = np.linspace(0, image_size, xy_res)
samplespace_z = np.linspace(0, 100, slices)


polarizer = np.array([1.0, 0.0])
analyzer = np.array([0.0, 1.0])


phiord = compute_phiord(ordinary_n, resolution=delta_z)

@jit(nopython=True)
def mat_mult(jm,njm):
    prod = np.zeros((2,2),dtype=np.complex128)
    prod[0,0] = jm[0][0]*njm[0][0]+jm[0][1]*njm[1][0]
    prod[0,1] = jm[0][0]*njm[0][1]+jm[0][1]*njm[1][1]
    prod[1,0] = jm[1][0]*njm[0][0]+jm[1][1]*njm[1][0]
    prod[1,1] = jm[1][0]*njm[0][1]+jm[1][1]*njm[1][1]
    return prod


#radial droplet for testing purposes
@jit(nopython=True)
def gamma (x,y):
    return np.arctan2(x, y)

@jit(nopython=True)
def beta (x,y,z):
    return np.arctan2(z, np.sqrt(x**2 + y**2))
    
@jit(nopython=True)
def _jm_funcs(betafunc,gammafunc,**kwargs):
    """
    should everything be normalized to +/-1?
    """
    opts = {}
    opts.update(kwargs)
    jmimage = np.zeros((xy_res, xy_res, 2, 2))
    ssx = np.linspace(-1,1,opts['xres'])
    ssy = np.linspace(-1,1,opts['yres'])
    ssz = np.linspace(-1,1,opts['zres'])
    for i, x in np.ndenumerate(ssx):
        for j, y in np.ndenumerate(ssy):
            jm = np.identity(2)
            for z in ssz:
                phiex = compute_phiex(betafunc(x,y,z),
                    ordinary_n,extraordinary_n,resolution=delta_z)
                njm =jones_matrix(gammafunc(x,y,z), phiex, phiord)
                jm=mat_mult(jm,njm)
            jmimage[i,j]=jm
    return jmimage

#todo: finish entry point, options handlers, 
def compute_jmimage(betafunc,gammafunc,**kwargs):
    """
    Subject to assumptions (see below), computes an image of Jones Matrices
    from functions that define a directorfield or from precomputed data.
    Director field is parameterized with cartesian coordinates and specified
    by angles beta and gamma. Recall, the director is a unitlength headless
    vector.
    """
    opts={'xres':300,
          'yres':300,
          'zres':600,
          'ne':1.7,
          'no':1.6,
          'wavlen':650,
          'beta_sym':None,
          'gamm_sym':None,
          'interp':False,
          'bounds':None}
    opts.update(kwargs)
    if (inspect.isfunction(betafunc) and inspect.isfunction(gammafunc)):
        #check num args in each
        if (len(inspect.getargspec(betafunc)[0])<3 or len(inspect.getargspec(gammafunc)[0])<3):
            #if fewer than 3, need symmetries opts
            pass
        elif (len(inspect.getargspec(betafunc)[0])>3 or len(inspect.getargspec(gammafunc)[0])>3):
            #too many args
            raise ValueError('Bad beta and gamm functions')
        else:
            #compute jmimage by functions with opts
            jmimage = _jm_funcs(betafunc,gammafunc,opts)
            pass
        pass
    else:
        #we've got data!
        pass
    return jmimage



#outline of actual jmimage computation engine
#will be more complicated due to uncertain beta/gamma symmetry axes
#need to check data for objects e.g. colloids, walls.
#todo for jit, don't use zip, check np.ndenumerate or ndindex
#also, may need to figure out function calls
#@jit(nopython = True)
"""
for i, x in np.ndenumerate(samplespace_x - droplet_center_x):
    print(i)
    if np.abs(x) < droplet_r:
        for j, y in np.ndenumerate(samplespace_y - droplet_center_y):
            if np.sqrt(np.abs(y)**2+np.abs(x)**2) < droplet_r:
                jm = np.identity(2)
                gamma = np.arctan2(x, y)
                for z in samplespace_z:
                    if np.sqrt(np.abs(y)**2 +
                               np.abs(x)**2 +
                               np.abs(z)**2) < droplet_r:
                        beta = np.arctan2(z-droplet_center_z,
                                          np.sqrt(x**2 + y**2))
                        phiex = compute_phiex(beta,
                                              ordinary_n,
                                              extraordinary_n,
                                              resolution=delta_z)
                        jm = np.dot(jm, jones_matrix(gamma, phiex, phiord))
                jmimage[i, j] = jm
return jmimage
"""