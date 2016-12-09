from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(dexcription='File select and visualizer options')

parser.add_argument('-f', type=str, help='file handle of qtensor')
parser.add_argument('-k2',type=float,help='The k2 value to read')
parser.add_argument('-LdG24',type=float,help='The LdG24 value to read')

args = parser.parse_args()

LdG24 = '0.6'
k2str = '0.5'
data = np.genfromtxt('/Users/zoeydavidson/Documents/Code/Beller_Nematics/minion/Results/Qtensor_42x42x120_K2_'+k2str+'_K1_6.0_LdG24_'+LdG24+'.dat')
dataD = np.genfromtxt('/Users/zoeydavidson/Documents/Code/Beller_Nematics/minion/Results/Defects_K2_'+k2str+'_K1_6.0_LdG24_'+LdG24+'.dat')

xsize = 42
ysize = 42
zsize = 120


#Defect data
dataDx = dataD[:,0]
dataDy = dataD[:,1]
dataDz = dataD[:,2]

print np.shape(data)[0]

key = np.array([999, 0, 0, 1, 2]) #we just want to get rid of that 1 in nz

data2 = np.where(key == data, [0,0,0,0,0], data) #replace with zeros
print np.shape(data2[:,1])[0]

###########DATA is in column major order 

datax = np.reshape(data2[:,1],(xsize,ysize,zsize),order='F')
datay = np.reshape(data2[:,2],(xsize,ysize,zsize),order='F')
dataz = np.reshape(data2[:,3],(xsize,ysize,zsize),order='F')
#zerolist = np.reshape(np.zeros(np.shape(data)[0]),(xsize,ysize,zsize),order='F')

#send to quiver

print "max z = "+ np.str(np.max(dataz))
print "min z = "+ np.str(np.min(dataz))

# Plot
fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0)) # figure with white background
mlab.quiver3d(datax,datay,dataz,mode='cylinder',vmax=np.max(np.abs(dataz)),vmin=np.min(0),scalars=(np.abs(dataz)))

 
# prevent segfautl (malloc too large) on osx
vectors = fig.children[0].children[0].children[0]
vectors.glyph.mask_points.maximum_number_of_points = 8000 # "Manually" set Maximum number of points
vectors.glyph.mask_input_points = True # turn masking on
vectors.glyph.color_mode = 'color_by_scalar'
vectors.glyph.mask_points.on_ratio = 12
vectors.glyph.mask_points.random_mode = False

#to display defect points
mlab.points3d(dataDx,dataDy,dataDz, scale_factor=0.5)

mlab.show()

#for 2D slices
#plt.quiver(posx, posy, data2[:,1],data2[:,2])
#plt.quiver(datax[::10], datay[::10])
#plt.quiver(posx, posy, data2[:,1],data2[:,2],angles='xy')
#plt.show()

