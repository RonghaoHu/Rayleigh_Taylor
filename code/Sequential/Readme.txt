**********************************************************************
This is an inviscid hydrodynamics solver making use of HLL rieman solver.
Contributed by NYU graduate student:
Xiaoyi Xie, Yu Guo, Xinwei Li.
*********************************************************************
To make use of the code to do other similar hydrodynamics simulation.
One need to change init_water_oil_3d, and the domain variable
X, Y, Z, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX.

To change grid size, just change the macro variable at the beginning.
Default
#define X 32
#define Y 32
#define Z 96

To run the code.
1, Change the dimension X Y Z ( default 32 32 96 ) in seq.c
2, Run make Makefile to get the binary executable "seq"
3, ./seq

The program will generate 3d and 2d data every 0.05s.
The total simulation time is 15s. 
