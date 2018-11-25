from pylab import *
import numpy as np
import sys

Input= sys.argv[1]
Output= sys.argv[2]
nx = 200
ny = 2
nz = 200

data = loadtxt("%s"%Input)
density = data[:,0].reshape([nx,ny,nz])
bx = data[:,7].reshape([nx,ny,nz])
by = data[:,5].reshape([nx,ny,nz])
bz = data[:,6].reshape([nx,ny,nz])

#title('lalal')
contourf(density[:,0,:], 40, cmap='RdBu')
colorbar()
savefig("%s"%Output,dpi=800)

