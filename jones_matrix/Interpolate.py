"""
Created on August 13th, 2015
Take a matrix from LdG nematic simulation by Dan Beller and interpolate to
create a finer grid for use in Jones Matrix calculation
By: Lisa Tran (lisa.tran11@gmail.com)
"""

#we will revist this...
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# reshape the data arrays to allow for easy interpolated element insertion
shape = datax.shape #datax.shape = datay.shape = dataz.shape

onebyonex = [[] for i in range(shape[1]*shape[2])] # make an empty list of lists
onebyoney = [[] for i in range(shape[1]*shape[2])] # make an empty list of lists
onebyonez = [[] for i in range(shape[1]*shape[2])] # make an empty list of lists
# each list will contain a 1D row/column of the 3D data set

for j in range(shape[2]):
    for i in range(shape[1]):
        onebyonex[j*shape[2]+i] = datax[:,i,j]

for j in range(shape[2]):
    for i in range(shape[1]):
        onebyoney[j*shape[2]+i] = datay[:,i,j]

for j in range(shape[2]):
    for i in range(shape[1]):
        onebyonez[j*shape[2]+i] = dataz[:,i,j]

x = range(shape[0])
y = range(shape[1])
z = range(shape[2])

interpfuncx = RegularGridInterpolator( (x,y,z), datax )
interpfuncy = RegularGridInterpolator( (x,y,z), datay )
interpfuncz = RegularGridInterpolator( (x,y,z), dataz )

a = np.linspace(0, 1, int(Lx)*int(outputskiprods)*0.1)
# define desired interval; 0.1um=100nm spacing

# NOT AN INTUITIVE STOPPING CONDITION: should fix

np.delete(a, [0, int(Lx)*int(outputskiprods)*0.1-1],0)
# delete 0 and 1 from the list
pt = []
pts = [[] for i in range(shape[0]-1)]

# create a list with intervals included to call into interpfunc
for i in range(shape[0]-1):
    for j in a:
        pt = i+j
        pts[i] = np.append(pts[i],pt)
del(pt)

newdatax = np.zeros([size(pts[:][:]),size(pts[:][:]),size(pts[:][:])], dtype =
float)
# create empty array to values created by interpfunc

# note: size(pts[0][:]) = size(pts[1][:]) = size(pts[2][:]), etc.
for m in range((shape[0])-1):
    for n in range(size(pts[0][:])):
        for k in range((shape[0])-1):
            for p in range(size(pts[0][:])):
                for i in range(shape[0]-1):
                    print i # to see if the simulation is still going
                    for j in range(size(pts[0][:])):
                        newdatax[i*size(pts[0][:])+j, k*size(pts[0][:])+p,
                        m*size(pts[0][:])+n ] = interpfuncx([pts[i][j],
                        pts[k][p], pts[m][n]])
# this previous step takes an absurd amount of time to run... have to fix
# amount of time this would take to complete is unrealistic/impractical

"""
