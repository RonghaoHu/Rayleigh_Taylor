from pylab import *
import numpy as np
from mayavi import mlab
import sys

Input= sys.argv[1]
Output= sys.argv[2]

nx = 200
ny = 2
nz = 200

data = loadtxt("%s"%Input)
density = data[:,0].reshape([nz,nx,ny])
bz = data[:,7].reshape([nz,nx,ny])
bx = data[:,5].reshape([nz,nx,ny])
by = data[:,6].reshape([nz,nx,ny])
#mlab.contour3d(zz, contours=3, transparent=True)
mlab.pipeline.volume(mlab.pipeline.scalar_field(density),color=(1.,1.,1.))
src = mlab.pipeline.vector_field(bx, by, bz)
mlab.pipeline.vectors(src, colormap='plasma',mask_points=20, scale_factor=10., scale_mode='vector', resolution=20, mode='arrow')
#mlab.outline()
mlab.show()
