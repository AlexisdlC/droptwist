from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt



data = np.genfromtxt('/Users/zoeydavidson/Documents/Code/Beller_Nematics/nematicv81_allfiles/Results/Defects_chol1_918_Lz140.dat')

xsize = 42
ysize = 42
zsize = 140

print np.shape(data)[0]
print np.shape(data[0])

dataDx = data[:,0]
dataDy = data[:,1]
dataDz = data[:,2]


#datax = np.reshape(data[:,0],(xsize,ysize,zsize),order='F')
#datay = np.reshape(data[:,1],(xsize,ysize,zsize),order='F')
#dataz = np.reshape(data[:,2],(xsize,ysize,zsize),order='F')
#zerolist = np.reshape(np.zeros(np.shape(data)[0]),(xsize,ysize,zsize),order='F')

mlab.points3d(dataDx,dataDy,dataDz)

