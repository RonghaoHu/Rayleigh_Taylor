from pylab import *
import numpy as np
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

#title('lalal')
contourf(density[:,:,0], 40, cmap='RdBu')
colorbar()
savefig("%s"%Output,dpi=800)

