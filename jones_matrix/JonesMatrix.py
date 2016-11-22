# -*- coding: utf-8 -*-
"""
Jones Matrix returns a numpy array form
For light traveling along z, gamma is the angle of the director in the xy plane.
phiex has in it, beta, angle to z.
"""

import numpy as np
import numba as nb

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
