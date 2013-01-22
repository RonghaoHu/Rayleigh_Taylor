**********************************************************************
This is a parallel inviscid hydrodynamics solver making use of HLL rieman solver.
Contributed by NYU graduate student:
Xiaoyi Xie, Yu Guo, Xinwei Li.
*********************************************************************
To make use of the code to do other similar hydrodynamics simulation.
One need to change init_water_oil_3d initial configuration, and the domain variable
X, Y, Z, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX.

To change grid size, just change the macro variable at the beginning.
Default
#define X 32
#define Y 32
#define Z 96

To run the code, one need NVIDIA GPU hardware, CUDA Toolkit,
CUDA Driver installed on computer. And follow the steps below:

1, Change the dimension X Y Z ( default 32 32 96 ) in hydro3d.cu
2, Run make Makefile to get the binary executable "hydro3d"
3, ./hydro3d

"hydro3d.cu" is our final cuda code making use of shared memory. "hydro3d_global.cu"
is our test code making use of global memory which is two to three times slower than
"hydro3d.cu".

The program will generate 3d and 2d data every 0.05s.
The total simulation time is 15s. 