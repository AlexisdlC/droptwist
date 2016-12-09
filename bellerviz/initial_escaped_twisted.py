import numpy as np
import numba as nb

@nb.njit
def cot(x):
    """
    cotangent
    """
    return 1.0/np.tan(x)

@nb.njit
def arccot(x):
    """
    arccotangent (inverse cotangent)
    """
    return np.arctan(1.0/x)

@nb.njit
def beta1(k2,k24):
    return 180.0*arccot(np.sqrt(k2/(k24*(k24-2*k2))))/np.pi


@nb.njit
def beta(s,k2,k24):
    """
    s between 0 and 1.
    k2 = K2/K3
    k24 = K24/k3
    """
    _cotbeta1 = np.sqrt(k2/(k24*(k24-2*k2)))
    _s0 = np.sqrt(k2)*(_cotbeta1)+np.sqrt(k2*_cotbeta1**2+1)
    _nom = 2*np.sqrt(k2)*_s0*s
    _denom = _s0**2-s**2
    return np.arctan(_nom/_denom)

@nb.njit
def build_cap(xs,ys,zs,R,k2,k24):
    out_dat = np.zeros((xs,ys,zs,5))
    for z in np.arange(zs):
        for y in np.arange(ys):
            for x in np.arange(xs):
                r = np.sqrt((x-xs/2.0)**2+(y-ys/2.0)**2)
                if r>np.sqrt(R**2):
                    out_dat[x,y,z,0]=999
                    out_dat[x,y,z,3]=1
                    out_dat[x,y,z,4]=2
                elif r<np.sqrt(R**2):
                    out_dat[x,y,z,0]=1.0
                    b = beta(1.0*r/R,k2,k24)
                    out_dat[x,y,z,0]=1.0
                    out_dat[x,y,z,1]=-np.sin(np.arctan2((y-ys/2.0),
                                            (x-xs/2.0)))*np.sin(b)
                    out_dat[x,y,z,2]=np.cos(np.arctan2((y-ys/2.0),
                                            (x-xs/2.0)))*np.sin(b)
                    out_dat[x,y,z,3]=np.cos(b)
                    out_dat[x,y,z,4]=0
    return out_dat

def to_qtensor(arr):
    s = np.ravel(arr[:,:,:,0],order='F')
    x = np.ravel(arr[:,:,:,1],order='F')
    y = np.ravel(arr[:,:,:,2],order='F')
    z = np.ravel(arr[:,:,:,3],order='F')
    b = np.ravel(arr[:,:,:,4],order='F')
    return np.vstack((s,x,y,z,b)).T
