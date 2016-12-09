from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt



LdG24 = '0.6' #the qtensor filenames print with certain extra info in them, these must match
k2str = '0.6' #ditto
data = np.genfromtxt('/Users/zoeydavidson/Documents/Code/Beller_Nematics/nematicv81_allfiles/Results/Qtensor_42x42x120_K2_'+k2str+'_K1_6.0initq0.dat')
dataD = np.genfromtxt('/Users/zoeydavidson/Documents/Code/Beller_Nematics/nematicv81_allfiles/Results/Defects_K2_'+k2str+'_K1_6.0initq0.dat')


xsize = 42 #sizes must match simulation
ysize = 42
zsize = 120


#Defect data from defect file
dataDx = dataD[:,0]
dataDy = dataD[:,1]
dataDz = dataD[:,2]

print np.shape(data)[0] #just checking

key = np.array([999, 0, 0, 1, 2]) #we just want to get rid of that 1 in nz # 999 means empty voxel

data2 = np.where(key == data, [0,0,0,0,0], data) #replace 999 rows with zero rows
print np.shape(data2[:,1])[0]

###########DATA is in column major order, specified by order='F'
datax = np.reshape(data2[:,1],(xsize,ysize,zsize),order='F')
datay = np.reshape(data2[:,2],(xsize,ysize,zsize),order='F')
dataz = np.reshape(data2[:,3],(xsize,ysize,zsize),order='F')


def draw_d_field(data):
    #send to quiver
    # Plot
    fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0)) # figure with white background
    mlab.quiver3d(datax,datay,dataz,mode='cylinder',vmax=np.max(np.abs(dataz)),vmin=np.min(0),scalars=(np.abs(dataz)))


    # prevent segfautl (malloc too large) on osx

    #vectors = fig.children[0].children[0].children[0]
    vectors = mlab.quiver3d(dat[1],dat[2],dat[3],mode='cylinder',vmax=np.max(np.abs(dat[3])),vmin=np.min(0),scalars=(np.abs(dat[3])))
    vectors.glyph.mask_points.maximum_number_of_points = 8000 # "Manually" set Maximum number of points
    vectors.glyph.mask_input_points = True # turn masking on
    vectors.glyph.color_mode = 'color_by_scalar'
    vectors.glyph.mask_points.on_ratio = 12
    vectors.glyph.mask_points.random_mode = False

    #to display defect points
    mlab.points3d(dataDx,dataDy,dataDz, scale_factor=0.5)

    mlab.show()
