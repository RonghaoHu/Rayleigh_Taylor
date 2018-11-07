import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()
data1 = np.loadtxt("C:\density_32_2_96_2d.dat")
data2 = np.loadtxt("C:\density_64_2_192_2d.dat")
data3 = np.loadtxt("C:\density_128_2_384_2d.dat")
data4 = np.loadtxt("C:\density_256_2_768_2d.dat")

ax1=fig.add_subplot(1,4,1)
cax = ax1.contourf(data1, 40)
ax2=fig.add_subplot(1,4,2)
ax2.contourf(data2, 40)
ax3=fig.add_subplot(1,4,3)
ax3.contourf(data3, 40)
ax4=fig.add_subplot(1,4,4)
ax4.contourf(data4, 40)

plt.colorbar(cax)
plt.show()
