# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:30:56 2014

@author: zoeydavidson
"""

import numpy as np
import numba as nb

@nb.njit(nogil=True)
def compute_phiex(beta,
                  ordinary_n,
                  extraordinary_n,
                  wavelen,
                  resolution):
    """
    Returns the extraordinary phase shift caused by a field.
    Computations in microns.
    beta is the angle of the director to the optical (z) axis.
    """
    _nExtraBeta_num = ordinary_n*extraordinary_n
    _nExtraBeta = _nExtraBeta_num /
                   (np.sqrt((np.sin(beta)*ordinary_n)**2 +
                     (np.cos(beta)*extraordinary_n)**2))
    return 2*np.pi*np.sqrt(_nExtraBeta)*resolution/wavelen


# A the ordinary phase component is constant.
@nb.njit(nogil=True)
def compute_phiord(ordinary_n, wavelen, resolution):
    """
    Returns the ordinary phase shift caused by a field.
    Computations in microns.
    """
    return 2*np.pi*ordinary_n*resolution/wavelen
