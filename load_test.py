import numpy as np
from bellerviz import load_qtensor as lq

%gui qt
from mayavi import mlab

fpath = '/Users/zoeydavidson/Documents/Code/lcviz/Results/'
dat = lq.load_dfield('cap_919_handbuilt_relax2',fpath)

fpath2 = '/Users/zoeydavidson/Documents/Code/lcviz/'
dat = lq.load_dfield('/Users/zoeydavidson/Documents/Code/lcviz/Qtensor_202x202x3_cap_919_handbuilt.dat',Lx=202,Ly=202,Lz=3,outputskiprods=0,rel=False,path='')

fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0)) # figure with white background

vectors = mlab.quiver3d(dat[1],dat[2],dat[3],mode='cylinder',
            vmax=np.max(np.abs(dat[3])),vmin=np.min(0),
            scalars=(np.abs(dat[3])))
vectors.glyph.mask_points.maximum_number_of_points = 8000 # "Manually" set Maximum number of points
vectors.glyph.mask_input_points = True # turn masking on
vectors.glyph.color_mode = 'color_by_scalar'
vectors.glyph.mask_points.on_ratio = 12
vectors.glyph.mask_points.random_mode = False
