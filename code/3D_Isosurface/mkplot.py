from pylab import *
import numpy as np
from mayavi import mlab
import sys

Input= sys.argv[1]
Output= sys.argv[2]

nx = 100
ny = 50
nz = 50

data = loadtxt("%s"%Input)
density = data[:,0].reshape([nx,ny,nz])
bx = data[:,7].reshape([nx,ny,nz])
by = data[:,5].reshape([nx,ny,nz])
bz = data[:,6].reshape([nx,ny,nz])
#contourf(z[:,:,25], 40, cmap='RdBu')
#colorbar()
#show()
#mlab.contour3d(zz, contours=3, transparent=True)
mlab.pipeline.volume(mlab.pipeline.scalar_field(density),color=(1.,1.,1.))
src = mlab.pipeline.vector_field(bx, by, bz)
mlab.pipeline.vectors(src, colormap='plasma',mask_points=20, scale_factor=10., scale_mode='vector', resolution=20, mode='arrow')
#mlab.outline()
mlab.show()
