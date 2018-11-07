from pylab import *
import numpy as np
from mayavi import mlab
import sys

Input= sys.argv[1]
Output= sys.argv[2]

data = loadtxt("%s"%Input)
z = data[:,0]
z = z.reshape([100,50,50])
#contourf(z[:,:,0], 40)
#colorbar()
#show()
#mlab.contour3d(zz, contours=3, transparent=True)
mlab.pipeline.volume(mlab.pipeline.scalar_field(z))
mlab.show()
