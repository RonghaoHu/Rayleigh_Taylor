from pylab import *
import numpy as np
import sys

Input= sys.argv[1]
Output= sys.argv[2]

data = loadtxt("%s"%Input)
z = data
#title('lalal')
contourf(z, 40)
colorbar()
savefig("%s"%Output,dpi=800)

