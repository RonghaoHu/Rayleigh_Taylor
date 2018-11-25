// -*- mode: C -*-

#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include "timing.h"
#define PI    3.14159265
#define Z_ion 1.0
#define A_ion 2.0
#define G     6.e13
#define GAMMA 1.666667
#ifdef RTI2D
  #define X 200
  #define Y 2
  #define Z 200
#else
#define X 50
#define Y 50
#define Z 100
#endif
#define XMIN -0.0001
#define XMAX 0.0001
#define YMIN -0.0001
#define YMAX 0.0001
#define ZMIN -0.0006
#define ZMAX 0.0006
#define THETA 2.0
#define tmax 15.e-9

#define BLOCK_SIZE 256
__constant__ float P0 = 1.2e16;
__constant__ float rhol = 33.e3;
__constant__ float rhoh = 66.e3;


void Grid(float *gridX, float *gridY, float *gridZ) {
  *gridX = (XMAX - XMIN) / (X - 1);
  *gridY = (YMAX - YMIN) / (Y - 1);
  *gridZ = (ZMAX - ZMIN) / (Z - 1);
}

__device__ void d_Grid(float* gridX, float* gridY, float* gridZ){
  *gridX = (XMAX - XMIN) / (X - 1);
  *gridY = (YMAX - YMIN) / (Y - 1);
  *gridZ = (ZMAX - ZMIN) / (Z - 1);
}

__device__ float GradientX(float *phys, int N, int L, int O, float dx) {
  if (N < L*Y*Z) {
    return (phys[N+L*Y*Z+O]-phys[N+(X-1)*L*Y*Z+O])/(2*dx);
  }
  else if (N >= (X-1)*L*Y*Z) {
    return (phys[N-(X-1)*L*Y*Z+O]-phys[N-L*Y*Z+O])/(2*dx);
  }
  else {
    return (phys[N+L*Y*Z+O]-phys[N-L*Y*Z+O])/(2*dx);
  }
}

__device__ float GradientY(float *phys, int N, int L, int O, float dy) {
  if (N%(L*Y*Z) < L*Z) {
    return (phys[N+L*Z+O]-phys[N+(Y-1)*L*Z+O])/(2*dy);
  }
  else if (N%(L*Y*Z) >= (Y-1)*L*Z) {
    return (phys[N-(Y-1)*L*Z+O]-phys[N-L*Z+O])/(2*dy);
  }
  else {
    return (phys[N+L*Z+O]-phys[N-L*Z+O])/(2*dy);
  }
}

__device__ float GradientZ(float *phys, int N, int L, int O, float dz) {
  return (phys[N+L+O]-phys[N-L+O])/(2*dz);
}

__global__ void h_Ucalc(float *U, float *phys, int NThreads) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float local_phys[BLOCK_SIZE*8];
  
  if( i < NThreads){
    for(l = 0; l < 8; l++) local_phys[8*threadIdx.x + l] = phys[8*i + l];
    U[8*i+0] = local_phys[8*threadIdx.x+0];
    U[8*i+1] = local_phys[8*threadIdx.x+0] * local_phys[8*threadIdx.x+1];
    U[8*i+2] = local_phys[8*threadIdx.x+0] * local_phys[8*threadIdx.x+2];
    U[8*i+3] = local_phys[8*threadIdx.x+0] * local_phys[8*threadIdx.x+3];
    U[8*i+4] = local_phys[8*threadIdx.x+4] / (GAMMA - 1) + 0.5 * local_phys[8*threadIdx.x+0] * (local_phys[8*threadIdx.x+1] * local_phys[8*threadIdx.x+1] + local_phys[8*threadIdx.x+2] * local_phys[8*threadIdx.x+2] + local_phys[8*threadIdx.x+3] * local_phys[8*threadIdx.x+3]);
    U[8*i+5] = local_phys[8*threadIdx.x+5];
    U[8*i+6] = local_phys[8*threadIdx.x+6];
    U[8*i+7] = local_phys[8*threadIdx.x+7];
  }
}

__global__ void h_Ucalcinv(float *phys, float *U, int NThreads) {     // phys[] = rho, v, p
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float local_U[BLOCK_SIZE*8];
  if( i < NThreads ) {
    for(l = 0; l < 8; l++) local_U[8*threadIdx.x + l] = U[8*i + l];
    phys[8*i+0] = local_U[8*threadIdx.x+0];
    phys[8*i+1] = local_U[8*threadIdx.x+1] / local_U[8*threadIdx.x+0];
    phys[8*i+2] = local_U[8*threadIdx.x+2] / local_U[8*threadIdx.x+0];
    phys[8*i+3] = local_U[8*threadIdx.x+3] / local_U[8*threadIdx.x+0];
    phys[8*i+4] = (local_U[8*threadIdx.x+4] - 0.5 * (local_U[8*threadIdx.x+1] * local_U[8*threadIdx.x+1] + local_U[8*threadIdx.x+2] * local_U[8*threadIdx.x+2] + local_U[8*threadIdx.x+3] * local_U[8*threadIdx.x+3])/ local_U[8*threadIdx.x+0]) * (GAMMA - 1);
    phys[8*i+5] = local_U[8*threadIdx.x+5];
    phys[8*i+6] = local_U[8*threadIdx.x+6];
    phys[8*i+7] = local_U[8*threadIdx.x+7];
  }
}

__global__ void h_FluxCalcPX(float *FLX, float *physL, float *FRX, float *physR, int NThreads)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float d_physL[BLOCK_SIZE*8];
  __shared__ float d_physR[BLOCK_SIZE*8];
  
  if(i < NThreads){
    for(l = 0; l < 8; l++) d_physL[8*threadIdx.x+l] = physL[8*i+l];
    for(l = 0; l < 8; l++) d_physR[8*threadIdx.x+l] = physR[8*i+l];
    
    FLX[8*i+0] = d_physL[8*threadIdx.x+0] * d_physL[8*threadIdx.x+1];
    FLX[8*i+1] = d_physL[8*threadIdx.x+0] * d_physL[8*threadIdx.x+1] * d_physL[8*threadIdx.x+1] + d_physL[8*threadIdx.x+4];
    FLX[8*i+2] = d_physL[8*threadIdx.x+0] * d_physL[8*threadIdx.x+1] * d_physL[8*threadIdx.x+2];
    FLX[8*i+3] = d_physL[8*threadIdx.x+0] * d_physL[8*threadIdx.x+1] * d_physL[8*threadIdx.x+3];
    FLX[8*i+4] = d_physL[8*threadIdx.x+1] *(d_physL[8*threadIdx.x+0] *(d_physL[8*threadIdx.x+1] * d_physL[8*threadIdx.x+1] + d_physL[8*threadIdx.x+2] * d_physL[8*threadIdx.x+2] + d_physL[8*threadIdx.x+3] * d_physL[8*threadIdx.x+3]) * .5 + d_physL[8*threadIdx.x+4] * GAMMA / (GAMMA - 1));

    FRX[8*i+0] = d_physR[8*threadIdx.x+0] * d_physR[8*threadIdx.x+1];
    FRX[8*i+1] = d_physR[8*threadIdx.x+0] * d_physR[8*threadIdx.x+1] * d_physR[8*threadIdx.x+1] + d_physR[8*threadIdx.x+4];
    FRX[8*i+2] = d_physR[8*threadIdx.x+0] * d_physR[8*threadIdx.x+1] * d_physR[8*threadIdx.x+2];
    FRX[8*i+3] = d_physR[8*threadIdx.x+0] * d_physR[8*threadIdx.x+1] * d_physR[8*threadIdx.x+3];
    FRX[8*i+4] = d_physR[8*threadIdx.x+1] *(d_physR[8*threadIdx.x+0] *(d_physR[8*threadIdx.x+1] * d_physR[8*threadIdx.x+1] + d_physR[8*threadIdx.x+2] * d_physR[8*threadIdx.x+2] + d_physR[8*threadIdx.x+3] * d_physR[8*threadIdx.x+3]) * .5 + d_physR[8*threadIdx.x+4] * GAMMA / (GAMMA - 1));
  }
}


__global__ void h_FluxCalcPX_N(float *FL, float *phys, int NThreads)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float d_phys[BLOCK_SIZE*8];
  
  if(i < NThreads){
    for(l = 0; l < 8; l++) d_phys[8*threadIdx.x+l] = phys[8*i+l];
    
    FL[8*i+0] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+1];
    FL[8*i+1] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+1] + d_phys[8*threadIdx.x+4];
    FL[8*i+2] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+2];
    FL[8*i+3] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+3];
    FL[8*i+4] = d_phys[8*threadIdx.x+1] *(d_phys[8*threadIdx.x+0] *(d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+1] + d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+2] + d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+3]) * .5 + d_phys[8*threadIdx.x+4] * GAMMA / (GAMMA - 1));
    FL[8*i+5] = 0.0;
    FL[8*i+6] = d_phys[8*threadIdx.x+1]*d_phys[8*threadIdx.x+6]-d_phys[8*threadIdx.x+2]*d_phys[8*threadIdx.x+5];
    FL[8*i+7] = d_phys[8*threadIdx.x+1]*d_phys[8*threadIdx.x+7]-d_phys[8*threadIdx.x+3]*d_phys[8*threadIdx.x+5];
  }
}



__global__ void h_FluxCalcPY_N(float *FL, float *phys, int NThreads)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float d_phys[BLOCK_SIZE*8];

  if(i < NThreads){
    for(l = 0; l < 8; l++) d_phys[8*threadIdx.x+l] = phys[8*i+l];

    FL[8*i+0] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+2];
    FL[8*i+1] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+1];
    FL[8*i+2] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+2] + d_phys[8*threadIdx.x+4];
    FL[8*i+3] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+3];
    FL[8*i+4] = d_phys[8*threadIdx.x+2] *(d_phys[8*threadIdx.x+0] *(d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+1] + d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+2] + d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+3]) * .5 + d_phys[8*threadIdx.x+4] * GAMMA / (GAMMA - 1));
    FL[8*i+5] = d_phys[8*threadIdx.x+2]*d_phys[8*threadIdx.x+5]-d_phys[8*threadIdx.x+1]*d_phys[8*threadIdx.x+6];
    FL[8*i+6] = 0.0;
    FL[8*i+7] = d_phys[8*threadIdx.x+2]*d_phys[8*threadIdx.x+7]-d_phys[8*threadIdx.x+3]*d_phys[8*threadIdx.x+6];
  }
}


__global__ void h_FluxCalcPZ_N(float *FL, float *phys, int NThreads)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l;
  __shared__ float d_phys[BLOCK_SIZE*8];

  if(i < NThreads){
    for(l = 0; l < 8; l++) d_phys[8*threadIdx.x+l] = phys[8*i+l];
    FL[8*i+0] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+3];
    FL[8*i+1] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+1];
    FL[8*i+2] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+2];
    FL[8*i+3] = d_phys[8*threadIdx.x+0] * d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+3] + d_phys[8*threadIdx.x+4];
    FL[8*i+4] = d_phys[8*threadIdx.x+3] *(d_phys[8*threadIdx.x+0] *(d_phys[8*threadIdx.x+1] * d_phys[8*threadIdx.x+1] + d_phys[8*threadIdx.x+2] * d_phys[8*threadIdx.x+2] + d_phys[8*threadIdx.x+3] * d_phys[8*threadIdx.x+3]) * .5 + d_phys[8*threadIdx.x+4] * GAMMA / (GAMMA - 1));
    FL[8*i+5] = d_phys[8*threadIdx.x+3]*d_phys[8*threadIdx.x+5]-d_phys[8*threadIdx.x+1]*d_phys[8*threadIdx.x+7];
    FL[8*i+6] = d_phys[8*threadIdx.x+3]*d_phys[8*threadIdx.x+6]-d_phys[8*threadIdx.x+2]*d_phys[8*threadIdx.x+7];
    FL[8*i+7] = 0.0;
  }
}


void Ucalc(float *U, float *phys, int N) {     // phys[] = rho, vx, vy, vz, p
  int i=0;
  for (i=0;i<N;i++) {
    U[8*i+0] = phys[8*i+0];
    U[8*i+1] = phys[8*i+0] * phys[8*i+1];
    U[8*i+2] = phys[8*i+0] * phys[8*i+2];
    U[8*i+3] = phys[8*i+0] * phys[8*i+3];
    U[8*i+4] = phys[8*i+4] / (GAMMA - 1) + 0.5 * phys[8*i+0] * (phys[8*i+1] * phys[8*i+1] + phys[8*i+2] * phys[8*i+2] + phys[8*i+3] * phys[8*i+3]);
    U[8*i+5] = phys[8*i+5];
    U[8*i+6] = phys[8*i+6];
    U[8*i+7] = phys[8*i+7];
  }
}


void Ucalcinv(float *phys, float *U, int N) {     // phys[] = rho, v, p
  int i=0;
  for (i=0;i<N;i++) {
    phys[8*i+0] = U[8*i+0];
    phys[8*i+1] = U[8*i+1] / U[8*i+0];
    phys[8*i+2] = U[8*i+2] / U[8*i+0];
    phys[8*i+3] = U[8*i+3] / U[8*i+0];
    phys[8*i+4] = (U[8*i+4] - 0.5 * (U[8*i+1] * U[8*i+1] + U[8*i+2] * U[8*i+2] + U[8*i+3] * U[8*i+3])/ U[8*i+0]) * (GAMMA - 1);
    phys[8*i+5] = U[8*i+5];
    phys[8*i+6] = U[8*i+6];
    phys[8*i+7] = U[8*i+7];
  }
}


__global__ void init_conditions(float *physical, int NCell)
{
  int i=0, j=0, k=0, N=0;
  float dx, dy, dz;
  float x, y, z;
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  double scaleLen = (ZMAX-ZMIN)/20, c_pert;
  d_Grid(&dx,&dy,&dz);
  
  if( threadID < NCell){

    i = threadID/(Y*Z);
    j = threadID/Z - i*Y;
    k = threadID - (threadID/Z)*Z;
    x = XMIN + dx * i;
    y = YMIN + dy * j;
    z = ZMIN + dz * k;
    N = 8*threadID;

    physical[N+0] = (rhol + rhoh) / 2 + tanh((z)/scaleLen) * (rhoh - rhol) / 2;
    physical[N+1] = 0.0;
    physical[N+2] = 0.0;
    physical[N+3] = 0.0;
    physical[N+3] = 0.0;
#ifdef RTI2D
    c_pert = 1.0 - 0.05 * (1 + cos(2 * PI * (x) / (XMAX - XMIN))) * exp(-(z*z)/(scaleLen*scaleLen));
#else
    c_pert = 1.0 - 0.05 * (1 + cos(2 * PI * (x) / (XMAX - XMIN))) * (1 + cos(2 * PI * (y) / (YMAX - YMIN))) * exp(-(z*z)/(scaleLen*scaleLen));
#endif
    physical[N+4] = c_pert * (P0 - G * ((z) * (rhol + rhoh) / 2 + log(cosh((z)/scaleLen))*scaleLen*(rhoh - rhol) / 2));
    physical[N+5] = 0.0;
    physical[N+6] = 0.0;
    physical[N+7] = 0.0;
    //if (z < 0.0) {
    //  physical[N+0] = rhol;
    //  physical[N+1] = 0.0;
    //  physical[N+2] = 0.0;
    //  physical[N+3] = -0.01*(1. + cosf(2*PI*x/((XMAX)-(XMIN))))*(1. + cosf(2*PI*y/((YMAX)-(YMIN))))*(1.+cosf(2*PI*z/((ZMAX)-(ZMIN))))/8.;
    //                
    //  //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*y))*(1. + cos(4*PI*z/3.))/8.;
    //  //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
    //  physical[N+4] = P0-rhol*G*z;
    //} else {
    //  physical[N+0] = rhoh;
    //  physical[N+1] = 0.0;
    //  physical[N+2] = 0.0;
    //  physical[N+3] = -0.01*(1. + cosf(2*PI*x/((XMAX)-(XMIN))))*(1. + cosf(2*PI*y/((YMAX)-(YMIN))))*(1.+cosf(2*PI*z/((ZMAX)-(ZMIN))))/8.;
    //  //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*y))*(1. + cos(4*PI*z/3.))/8.;
    //  //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
    //  physical[N+4] = P0-rhoh*G*(z);
    //}
  }
}

void output_print_all(float *file, char *name, int x, int y, int z, int n, int variable){
  int Sx, Sy, Sz, element;
  int i,j,k,N;
  FILE *fp;
  char command[120];
  char filename[120];
  Sx = variable*Y*Z;
  Sy = variable*Z;
  Sz = variable;

  printf("write: %s_\n",name);
  for(i=0 ; i < variable ; i++){
    sprintf(command,"rm %s_%d.dat",name,i);
    system(command);
  }
  
  for(element = 0; element < variable; element++){
    sprintf(filename, "%s_%d.dat", name, element);
    fp = fopen(filename, "a+");
    for(i=0; i<x; i++){
      for(j=0; j<y;j++){
	for(k=0; k<z;k++){
	  N = Sx*i + Sy*j + Sz*k + element;
	  fprintf(fp, "%d\t%d\t%d\t%f\n",i,j,k,file[N]);
	}
      }
    }
    fclose(fp);
  }
}

void output_print(float *file,char *name, int element, int x, int y, int z, int n, int variable){
  int Sx, Sy, Sz;
  int i,j,k,N;
  FILE *fp;
  char command[120];
  Sx = variable*Y*Z;
  Sy = variable*Z;
  Sz = variable;
  
  printf("write: %s\n",name);
  sprintf(command,"rm %s",name);
  system(command);
  fp = fopen(name, "a+");
  for(i=0; i<x; i++){
    for(j=0; j<y;j++){
      for(k=0; k<z;k++){
	N = Sx*i + Sy*j + Sz*k + element;
	fprintf(fp, "%d\t%d\t%d\t%f\n",i,j,k,file[N]);
      }
    }
  }
  fclose(fp);
}
  
void output_file2(float *U,float t){

  int i=0, j, k, N;
  float xstep, ystep, zstep;
  Grid  (&xstep, &ystep, &zstep);
  int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
  
  char command[160];
  char name[160];
  float *phys = (float*) malloc (8*X*Y*Z*sizeof(float));
  Ucalcinv(phys, U, (X)*(Y)*(Z));

  sprintf(command, "rm density_%d_%d_%d_2d.dat",X,Y,Z);
  system(command);
  sprintf(name,"density_%d_%d_%d_2d.dat",X,Y,Z);
  FILE *dens = fopen(name, "a+");
    
  sprintf(command, "rm velocity_%d_%d_%d_2d.dat",X,Y,Z);
  system(command);
  sprintf(name, "velocity_%d_%d_%d_2d.dat",X,Y,Z);

  FILE *vel = fopen(name,"a+");
    
  sprintf(command, "rm pressure_%d_%d_%d_2d.dat",X,Y,Z);
  system(command);
  sprintf(name, "pressure_%d_%d_%d_2d.dat",X,Y,Z);

  FILE *press = fopen(name,"a+");
  
  for (k=0; k<Z; k++) {
    for (i=0; i<X; i++) {
      j = Y/2;
      N = Sx*i+Sy*j+Sz*k;
      fprintf(dens , "%f\t",phys[N]);
      //fprintf(vel  , "%f\t",phys[N+3]);
      fprintf(vel  , "%e\t",phys[N+6]);
      fprintf(press, "%f\t",phys[N+4]);
    }
    fprintf(dens , "\n");
    fprintf(vel  , "\n");
    fprintf(press, "\n");
  }
  fclose(dens);
  fclose(vel);
  fclose(press);

  sprintf(command, "cp density_%d_%d_%d_2d.dat %dx%d/density/%f.dat",X,Y,Z,X,Z,t);
  system(command);
  sprintf(command, "cp velocity_%d_%d_%d_2d.dat %dx%d/velocity/%f.dat",X,Y,Z,X,Z,t);
  system(command);
  sprintf(command, "cp pressure_%d_%d_%d_2d.dat %dx%d/pressure/%f.dat",X,Y,Z,X,Z,t);
  system(command);

  free (phys);
}

void output_file3(float *U, float t){
    
    int i=0, j, k, l, N;
    float xstep, ystep, zstep;
    Grid  (&xstep, &ystep, &zstep);
    int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
    char command[120];
    char name[120];
    
    float *phys = (float*) malloc (8*X*Y*Z*sizeof(float));
    Ucalcinv(phys, U, (X)*(Y)*(Z));
    
    sprintf(command, "rm density_%d_%d_%d_3d.dat",X,Y,Z);
    system(command);
    sprintf(name,"density_%d_%d_%d_3d.dat",X,Y,Z);

    FILE *dens = fopen(name, "a+");
    
    sprintf(command, "rm velocity_%d_%d_%d_3d.dat",X,Y,Z);
    system(command);
    sprintf(name, "velocity_%d_%d_%d_3d.dat",X,Y,Z);

    FILE *vel = fopen(name,"a+");
    
    sprintf(command, "rm pressure_%d_%d_%d_3d.dat",X,Y,Z);
    system(command);
    sprintf(name, "pressure_%d_%d_%d_3d.dat",X,Y,Z);

    FILE *press = fopen(name,"a+");
    
    for (k=0; k<Z; k++) {
        for (i=0; i<X; i++) {
            for (j=0; j<Y; j++) {
                for (l=0; l<8; l++) {
                    N = Sx*i+Sy*j+Sz*k+l;
                    fprintf(dens , "%e\t",phys[N]);
                    //fprintf(vel  , "%d\t%d\t%d\t%f\n",i,j,k,phys[N+3]);
                    //fprintf(press, "%d\t%d\t%d\t%f\n",i,j,k,phys[N+4]);
                }
                fprintf(dens, "\n");
            }
        }
    }
    fclose(dens);
    fclose(vel);
    fclose(press);

    sprintf(command, "cp density_%d_%d_%d_3d.dat %dx%dx%d/density/%f.dat",X,Y,Z,X,Y,Z,t);
    system(command);
    sprintf(command, "cp velocity_%d_%d_%d_3d.dat %dx%dx%d/velocity/%f.dat",X,Y,Z,X,Y,Z,t);
    system(command);
    sprintf(command, "cp pressure_%d_%d_%d_3d.dat %dx%dx%d/pressure/%f.dat",X,Y,Z,X,Y,Z,t);
    system(command);

    free (phys);
}


void Creat_folder(){
  char command[120];
  sprintf(command, "rm -r %dx%dx%d",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/velocity",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/density",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/pressure",X,Y,Z);
  system(command);
  sprintf(command, "rm -r %dx%d",X,Z);
  sprintf(command, "mkdir %dx%d",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/velocity",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/density",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/pressure",X,Z);
  system(command);
}

void Check_CUDA_Error(const char *message)
{
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n",message, cudaGetErrorString(error));
    return;
  } else{
    fprintf(stderr,"SUCCESS: %s: %s\n",message, cudaGetErrorString(error));
    return;
  }
}

__device__ int d_sgn(float x) {
  float S;
  if(x > 0. ) S =  1.;
  else S = -1.;
  return S;
}

__device__ float d_fabs(float x) {// the fabs function callable from device function
  return x < 0 ? -x : x;
}

__device__ float  minmod(float x, float y, float z) {// The minmod function, described in the paper
  float min, M;
  min = d_fabs(x);
  if (d_fabs(y) < min) min = d_fabs(y);
  if (d_fabs(z) < min) min = d_fabs(z);
  M = .25 * d_fabs(d_sgn(x) + d_sgn(y)) * (d_sgn(x) + d_sgn(z)) * min;
  return M;
}

__global__ void h_BoundaryX(float * phys_temp, float * phys, int NThreads)
{
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
  int i, j, k, l, N;

  if(threadID < NThreads){
    i = threadID/((Y)*(Z));
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    
   
    for ( l=0; l<8; l++){
      N = Sx*i + Sy*j + Sz*k + l;
      if (i<2){
	phys_temp[N] = phys[N - 2*Sx + X*Sx];
      } else if (i > X+1){
	phys_temp[N] = phys[N - 2*Sx - X*Sx];
      } else {
	phys_temp[N] = phys[N - 2*Sx];
      }
    }
  }
}

__global__ void h_BoundaryY(float * phys_temp, float * phys, int NThreads)
{
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
  int Sx2 = 8*(Y+4)*Z, Sy2 = 8*Z, Sz2 = 8;
  int i, j, k, l, N;

  if(threadID < NThreads){
    i = threadID/((Y+4)*(Z));
    j = threadID/(Z) - i*(Y+4);
    k = threadID - (threadID/(Z))*(Z);
    
    for ( l=0; l<8; l++){
      N = Sx2*i + Sy2*j + Sz2*k + l;
      if (j<2){
	phys_temp[N] = phys[Sx*i + Sy*(j+Y-2) + Sz*k + l];
      } else if (j > Y+1){
	phys_temp[N] = phys[Sx*i + Sy*(j-Y-2) + Sz*k + l];
      } else {
	phys_temp[N] = phys[Sx*i + Sy*(j - 2) + Sz*k + l];
      }
    }
  }
}


__global__ void h_BoundaryZ(float * phys_temp, float * phys, int NThreads)
{
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
  int Sx2 = 8*Y*(Z+4), Sy2 = 8*(Z+4), Sz2 = 8;
  int i, j, k, l, N;

  if(threadID < NThreads){
    i = threadID/((Y)*(Z+4));
    j = threadID/(Z+4) - i*(Y);
    k = threadID - (threadID/(Z+4))*(Z+4);

    for ( l=0; l<8; l++){
      N = Sx2*i + Sy2*j + Sz2*k + l;
      if      (k==0) {
	    if    (l==3)  phys_temp[N] = -phys[Sx*i+Sy*j+Sz*(1)+l];
        else          phys_temp[N] = phys[Sx*i+Sy*j+Sz*(0)+l];
      }
      else if (k==1) {
        if    (l==3)  phys_temp[N] = -phys[Sx*i+Sy*j+Sz*(0)+l];
        else          phys_temp[N] = phys[Sx*i+Sy*j+Sz*(0)+l];
      }
      else if (k==Z+2) {
        if    (l==3)  phys_temp[N] = -phys[Sx*i+Sy*j+Sz*(Z-1)+l];
        else          phys_temp[N] = phys[Sx*i+Sy*j+Sz*(Z-1)+l];
      }
      else if (k==Z+3) {
        if    (l==3) phys_temp[N] = -phys[Sx*i+Sy*j+Sz*(Z-2)+l];
        else         phys_temp[N] = phys[Sx*i+Sy*j+Sz*(Z-1)+l];
      }
      else            phys_temp[N] = phys[Sx*i+Sy*j+Sz*(k-2)+l];
    }
  }
}


__global__ void h_StateX(float* physL, float* physR, float * phys_temp, int NTreads)
{
  int i,j,k,l,N;
  int Sx = 8*Y*Z, Sy = 8*Z, Sz = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  __shared__ float local_phys[BLOCK_SIZE*4];

  if( threadID < NTreads){
    i = threadID/((Y)*(Z));
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    
    for(l = 0; l < 8; l++){
      N = Sx*i + Sy*j + Sz*k + l;
      local_phys[4*threadIdx.x + 0] = phys_temp[N];
      local_phys[4*threadIdx.x + 1] = phys_temp[N+Sx];
      local_phys[4*threadIdx.x + 2] = phys_temp[N+2*Sx];
      local_phys[4*threadIdx.x + 3] = phys_temp[N+3*Sx];
      physL[N] = local_phys[4*threadIdx.x + 1]   + 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 1]   - local_phys[4*threadIdx.x + 0]),    0.5*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 0]),    THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]));
      physR[N] = local_phys[4*threadIdx.x + 2] - 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]), 0.5*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 1]), THETA*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 2]));
      __syncthreads();
    }
  }
}


__global__ void h_StateY(float* physL, float* physR, float * phys_temp, int NTreads)
{
  int i,j,k,l,N,N2;
  int Sx = 8*(Y+1)*Z, Sy = 8*Z, Sz = 8;
  int Sx2 = 8*(Y+4)*Z, Sy2 = 8*Z, Sz2 = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  __shared__ float local_phys[BLOCK_SIZE*4];
  
  if( threadID < NTreads){
    i = threadID/((Y+1)*(Z));
    j = threadID/(Z) - i*(Y+1);
    k = threadID - (threadID/(Z))*(Z);
    
    for(l = 0; l < 8; l++){
      N = Sx*i + Sy*j + Sz*k + l;
      N2 = Sx2*i + Sy2*j + Sz2*k + l;
      local_phys[4*threadIdx.x + 0] = phys_temp[N2];
      local_phys[4*threadIdx.x + 1] = phys_temp[N2+Sy2];
      local_phys[4*threadIdx.x + 2] = phys_temp[N2+2*Sy2];
      local_phys[4*threadIdx.x + 3] = phys_temp[N2+3*Sy2];

      physL[N] = local_phys[4*threadIdx.x + 1]   + 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 1]   - local_phys[4*threadIdx.x + 0]),     0.5*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 0]),     THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]));
      physR[N] = local_phys[4*threadIdx.x + 2] - 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]), 0.5*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 1]), THETA*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 2]));
      __syncthreads();
    }
  }
}


__global__ void h_StateZ(float* physL, float* physR, float* phys_temp, int NTreads)
{
  int i,j,k,l,N,N2;
  int Sx = 8*Y*(Z+1), Sy = 8*(Z+1), Sz = 8;
  int Sx2 = 8*Y*(Z+4), Sy2 = 8*(Z+4), Sz2 = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  __shared__ float local_phys[BLOCK_SIZE*4];

  if( threadID < NTreads){
    i = threadID/((Y)*(Z+1));
    j = threadID/(Z+1) - i*(Y);
    k = threadID - (threadID/(Z+1))*(Z+1);
    
    for(l = 0; l < 8; l++){
      N = Sx*i + Sy*j + Sz*k + l;
      N2 = Sx2*i + Sy2*j + Sz2*k + l;

      local_phys[4*threadIdx.x + 0] = phys_temp[N2];
      local_phys[4*threadIdx.x + 1] = phys_temp[N2+Sz2];
      local_phys[4*threadIdx.x + 2] = phys_temp[N2+2*Sz2];
      local_phys[4*threadIdx.x + 3] = phys_temp[N2+3*Sz2];

      physL[N] = local_phys[4*threadIdx.x + 1]   + 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 1]   - local_phys[4*threadIdx.x + 0]),     0.5*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 0]),     THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]));
      physR[N] = local_phys[4*threadIdx.x + 2] - 0.5 * minmod (THETA*(local_phys[4*threadIdx.x + 2] - local_phys[4*threadIdx.x + 1]), 0.5*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 1]), THETA*(local_phys[4*threadIdx.x + 3] - local_phys[4*threadIdx.x + 2]));
      __syncthreads();
    }
  }
}


__global__ void h_FluxMidX(float* F_mid, float* global_max, float* FL, float *FR, float *UL, float *UR, float *physL, float *physR, int NThreads)
{
  int i, j, k, l,N;
  float AlphaPlus,AlphaMinus,max;
  float SoundSpeedL, SoundSpeedR;
  int Sx = 8*(Y)*(Z), Sy = 8*(Z), Sz = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  if(threadID < NThreads){
    i = threadID/(Y*Z);
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    N = Sx*i + Sy*j + Sz*k;
    __shared__ float local_max[BLOCK_SIZE];
    /***** x from 0 to X-1 **********/
    AlphaPlus = 0.;
    AlphaMinus = 0.;
    max = 0.;
    SoundSpeedL = sqrtf (GAMMA * physL[N+4] / physL[N+0]);
    SoundSpeedR = sqrtf (GAMMA * physR[N+4] / physR[N+0]);
    if (AlphaPlus  <  physL[N+1] + SoundSpeedL ) AlphaPlus  =  physL[N+1] + SoundSpeedL;
    if (AlphaMinus < -physL[N+1] + SoundSpeedL ) AlphaMinus = -physL[N+1] + SoundSpeedL;
    if (AlphaPlus  <  physR[N+1] + SoundSpeedR ) AlphaPlus  =  physR[N+1] + SoundSpeedR;
    if (AlphaMinus < -physR[N+1] + SoundSpeedR ) AlphaMinus = -physR[N+1] + SoundSpeedR;
    for (l=0; l<8; l++) {
      N = Sx*i+Sy*j+Sz*k+l;
      F_mid[N] = (AlphaPlus*FL[N]+AlphaMinus*FR[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
    }
    
    if (max < AlphaPlus) max = AlphaPlus;
    if (max < AlphaMinus) max = AlphaMinus;
    local_max[threadIdx.x] = max;
    __syncthreads();
    volatile float *c = local_max;
    float temp;
    if(blockIdx.x < gridDim.x - 1){
      for(unsigned int s=blockDim.x/2; s>32; s>>=1){
	if(threadIdx.x < s){
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? c[threadIdx.x + s] : temp;
	}
	__syncthreads();
      }
      if(threadIdx.x < 32)
	{	
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 32])? c[threadIdx.x + 32] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 16])? c[threadIdx.x + 16] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 8])? c[threadIdx.x + 8] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 4])? c[threadIdx.x + 4] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 2])? c[threadIdx.x + 2] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 1])? c[threadIdx.x + 1] : temp;
	}
      __syncthreads();
      if(threadIdx.x == 0 ){
	global_max[blockIdx.x] = local_max[threadIdx.x];
      //      printf("FluxMidX: blockId: %d max: %f\n",blockIdx.x, local_max[threadIdx.x]);
      }
    } else {
	for(unsigned int s=(NThreads - (gridDim.x-1)*blockDim.x)/2; s>0; s>>=1){
	  if(threadIdx.x < s) {
	    temp = c[threadIdx.x];
	    c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? \
	      c[threadIdx.x + s] : temp;
	  }
	  __syncthreads();
	}
	if(threadIdx.x == 0){
	  temp = c[0];
	  c[0] = temp = (temp < c[NThreads - (gridDim.x - 1)*blockDim.x - 1]) ? \
	    c[NThreads - (gridDim.x - 1)*blockDim.x -1]: temp;
	  global_max[blockIdx.x] = local_max[0];
	}
    }
  }
}


__global__ void h_FluxMidY(float* F_mid, float* global_max, float* FL, float *FR, float *UL, float *UR, float *physL, float *physR, int NThreads)
{
  int i, j, k, l,N;
  float AlphaPlus,AlphaMinus,max;
  float SoundSpeedL, SoundSpeedR;
  int Sx = 8*(Y+1)*(Z), Sy = 8*(Z), Sz = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  if(threadID < NThreads){
    i = threadID/((Y+1)*Z);
    j = threadID/Z - i*(Y+1);
    k = threadID - (threadID/Z)*Z;
    N = Sx*i + Sy*j + Sz*k;
    __shared__ float local_max[BLOCK_SIZE];
    /***** x from 0 to X-1 **********/
    AlphaPlus = 0.;
    AlphaMinus = 0.;
    max = 0.;
    SoundSpeedL = sqrtf (GAMMA * physL[N+4] / physL[N+0]);
    SoundSpeedR = sqrtf (GAMMA * physR[N+4] / physR[N+0]);
    if (AlphaPlus  <  physL[N+2] + SoundSpeedL ) AlphaPlus  =  physL[N+2] + SoundSpeedL;
    if (AlphaMinus < -physL[N+2] + SoundSpeedL ) AlphaMinus = -physL[N+2] + SoundSpeedL;
    if (AlphaPlus  <  physR[N+2] + SoundSpeedR ) AlphaPlus  =  physR[N+2] + SoundSpeedR;
    if (AlphaMinus < -physR[N+2] + SoundSpeedR ) AlphaMinus = -physR[N+2] + SoundSpeedR;
    for (l=0; l<8; l++) {
      N = Sx*i+Sy*j+Sz*k+l;
      F_mid[N] = (AlphaPlus*FL[N]+AlphaMinus*FR[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
    }
      
    if (max < AlphaPlus) max = AlphaPlus;
    if (max < AlphaMinus) max = AlphaMinus;
    local_max[threadIdx.x] = max;
    __syncthreads();
    volatile float *c = local_max;
    float temp;
    if(blockIdx.x < gridDim.x - 1){
      for(unsigned int s=blockDim.x/2; s>32; s>>=1){
          if(threadIdx.x < s){
              temp = c[threadIdx.x];
              c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? c[threadIdx.x + s] : temp;
          }
          __syncthreads();
      }
      if(threadIdx.x < 32)
      {
          temp = c[threadIdx.x];
          c[threadIdx.x] = temp = (temp < c[threadIdx.x + 32])? c[threadIdx.x + 32] : temp;
          temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 16])? c[threadIdx.x + 16] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 8])? c[threadIdx.x + 8] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 4])? c[threadIdx.x + 4] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 2])? c[threadIdx.x + 2] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 1])? c[threadIdx.x + 1] : temp;
	}
      __syncthreads();
      if(threadIdx.x == 0 ){
	global_max[blockIdx.x] = local_max[0];
      //      printf("FluxMidX: blockId: %d max: %f\n",blockIdx.x, local_max[threadIdx.x]);
      }
    } else {
	for(unsigned int s=(NThreads - (gridDim.x-1)*blockDim.x)/2; s>0; s>>=1){
	  if(threadIdx.x < s) {
	    temp = c[threadIdx.x];
	    c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? \
	      c[threadIdx.x + s] : temp;
	  }
	  __syncthreads();
	}
	if(threadIdx.x == 0){
	  temp = c[0];
	  c[0] = temp = (temp < c[NThreads - (gridDim.x - 1)*blockDim.x - 1]) ? \
	    c[NThreads - (gridDim.x - 1)*blockDim.x -1]: temp;
	  global_max[blockIdx.x] = local_max[0];
	}
    }
  }
}


__global__ void h_FluxMidZ(float* F_mid, float* global_max, float* FL, float *FR, float *UL, float *UR, float *physL, float *physR, int NThreads)
{
  int i, j, k, l,N;
  float AlphaPlus,AlphaMinus,max;
  float SoundSpeedL, SoundSpeedR;
  int Sx = 8*(Y)*(Z+1), Sy = 8*(Z+1), Sz = 8;
  int threadID = blockDim.x*blockIdx.x + threadIdx.x;
  if(threadID < NThreads){
    i = threadID/(Y*(Z+1));
    j = threadID/(Z+1) - i*Y;
    k = threadID - (threadID/(Z+1))*(Z+1);
    N = Sx*i + Sy*j + Sz*k;
    __shared__ float local_max[BLOCK_SIZE];
    /***** x from 0 to X-1 **********/
    AlphaPlus = 0.;
    AlphaMinus = 0.;
    max = 0.;
    SoundSpeedL = sqrtf (GAMMA * physL[N+4] / physL[N+0]);
    SoundSpeedR = sqrtf (GAMMA * physR[N+4] / physR[N+0]);
    if (AlphaPlus  <  physL[N+3] + SoundSpeedL ) AlphaPlus  =  physL[N+3] + SoundSpeedL;
    if (AlphaMinus < -physL[N+3] + SoundSpeedL ) AlphaMinus = -physL[N+3] + SoundSpeedL;
    if (AlphaPlus  <  physR[N+3] + SoundSpeedR ) AlphaPlus  =  physR[N+3] + SoundSpeedR;
    if (AlphaMinus < -physR[N+3] + SoundSpeedR ) AlphaMinus = -physR[N+3] + SoundSpeedR;
    for (l=0; l<8; l++) {
      N = Sx*i+Sy*j+Sz*k+l;
      F_mid[N] = (AlphaPlus*FL[N]+AlphaMinus*FR[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
    }
    if (max < AlphaPlus) max = AlphaPlus;
    if (max < AlphaMinus) max = AlphaMinus;
    local_max[threadIdx.x] = max;
    __syncthreads();
    volatile float *c = local_max;
    float temp;
    if(blockIdx.x < gridDim.x - 1){
      for(unsigned int s=blockDim.x/2; s>32; s>>=1){
          if(threadIdx.x < s){
              temp = c[threadIdx.x];
              c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? c[threadIdx.x + s] : temp;
          }
          __syncthreads();
      }
      if(threadIdx.x < 32)
      {
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 32])? c[threadIdx.x + 32] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 16])? c[threadIdx.x + 16] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 8])? c[threadIdx.x + 8] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 4])? c[threadIdx.x + 4] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 2])? c[threadIdx.x + 2] : temp;
	  temp = c[threadIdx.x];
	  c[threadIdx.x] = temp = (temp < c[threadIdx.x + 1])? c[threadIdx.x + 1] : temp;
      }
      __syncthreads();
      if(threadIdx.x == 0 ){
          global_max[blockIdx.x] = local_max[0];
      //      printf("FluxMidX: blockId: %d max: %f\n",blockIdx.x, local_max[threadIdx.x]);
      }
    } else {
	for(unsigned int s=(NThreads - (gridDim.x-1)*blockDim.x)/2; s>0; s>>=1){
	  if(threadIdx.x < s) {
	    temp = c[threadIdx.x];
	    c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? \
	      c[threadIdx.x + s] : temp;
	  }
	  __syncthreads();
	}
	if(threadIdx.x == 0){
	  temp = c[0];
	  c[0] = temp = (temp < c[NThreads - (gridDim.x - 1)*blockDim.x - 1]) ? \
	    c[NThreads - (gridDim.x - 1)*blockDim.x -1]: temp;
	  global_max[blockIdx.x] = local_max[0];
	}
    }
  }
}

__global__ void h_Max(float *out, float *in, int len)
{
  __shared__ float local_max[BLOCK_SIZE];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if( i < len)
    {
      local_max[threadIdx.x] = in[i];
      //printf("%d value: %f\n", i, in[i]);
    }
  __syncthreads();
  volatile float *c = local_max;
  float temp;
  if(blockIdx.x < gridDim.x - 1){
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
      if(threadIdx.x < s ){
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? c[threadIdx.x + s] : temp;
      }
      __syncthreads();
    }
    if(threadIdx.x < 32 )
      {	
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 32])? c[threadIdx.x + 32] : temp;
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 16])? c[threadIdx.x + 16] : temp;
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 8])? c[threadIdx.x + 8] : temp;
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 4])? c[threadIdx.x + 4] : temp;
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 2])? c[threadIdx.x + 2] : temp;
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + 1])? c[threadIdx.x + 1] : temp;
      }
    __syncthreads();
    if(threadIdx.x == 0 ){
      out[blockIdx.x] = local_max[0];
      //      printf("Max: blockID: %d local Max: %f\n",blockIdx.x, local_max[threadIdx.x]);
    }
  } else {
    for(unsigned int s=(len - (gridDim.x-1)*blockDim.x)/2; s>0; s>>=1){
      if(threadIdx.x < s) {
	temp = c[threadIdx.x];
	c[threadIdx.x] = temp = (temp < c[threadIdx.x + s])? c[threadIdx.x + s] : temp;
	//	printf("threadIdx: %d s: %d ID: %d value: %f\n",threadIdx.x, s, i, c[threadIdx.x]);
      }
      __syncthreads();
    }
    if(threadIdx.x == 0 ){
      temp = c[0];
      c[0] = temp = (temp < c[len - (gridDim.x - 1)*blockDim.x -1])? \
	c[len - (gridDim.x - 1)*blockDim.x -1]: temp;
      out[blockIdx.x] = local_max[0];
      //printf("Max: Last blockID: %d local Max: %f\n",blockIdx.x, local_max[threadIdx.x]);
    }
  }
}
  

void riemansolverX(float *F_mid, float *U, float *max)
{
  
  int threadsPerBlock, blocksPerGrid;
  /*************Ucalcinv********/
  float* phys; 
  cudaMalloc(&phys, 8*(X)*(Y)*(Z)*sizeof(float));
  threadsPerBlock = BLOCK_SIZE;
  blocksPerGrid =    				\
    ((X)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_Ucalcinv<<<blocksPerGrid, threadsPerBlock>>>(phys,U,(X)*(Y)*(Z));
  /*******************************/
  /*************Set_Boundary******************/
  float *phys_temp; 
  cudaMalloc(&phys_temp, 8*(X+4)*(Y)*(Z)*sizeof(float));

  blocksPerGrid =						\
    ((X+4)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_BoundaryX<<<blocksPerGrid, threadsPerBlock>>>(phys_temp, phys, (X+4)*(Y)*(Z));
  /*********************************/
  
  cudaFree(phys);

  /************Set_physState********************/
  float *physL;
  cudaMalloc(&physL, 8*(X+1)*(Y)*(Z)*sizeof(float));
  float *physR;
  cudaMalloc(&physR, 8*(X+1)*(Y)*(Z)*sizeof(float));

  blocksPerGrid =						\
    ((X+1)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_StateX<<<blocksPerGrid, threadsPerBlock>>>(physL, physR, phys_temp,(X+1)*(Y)*(Z));
  //  Check_CUDA_Error("at riemansolverX: Set_PhysStateX");
  /*****************************/

  cudaFree(phys_temp);

  /***********Calc_Flux****************/
  float *FLX;
  cudaMalloc(&FLX, 8*(X+1)*(Y)*(Z)*sizeof(float));
  float *FRX;
  cudaMalloc(&FRX, 8*(X+1)*(Y)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X+1)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_FluxCalcPX_N<<<blocksPerGrid, threadsPerBlock>>>(FLX,  physL, (X+1)*(Y)*(Z));
  h_FluxCalcPX_N<<<blocksPerGrid, threadsPerBlock>>>(FRX,  physR, (X+1)*(Y)*(Z));
  
  /************************************/

  /********Calc U*****************/
  float* UL;
  cudaMalloc(&UL, 8*(X+1)*(Y)*(Z)*sizeof(float));
  float* UR;
  cudaMalloc(&UR, 8*(X+1)*(Y)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X+1)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;  
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UL, physL, (X+1)*(Y)*(Z));
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UR, physR, (X+1)*(Y)*(Z));
  /*******************************/

  /************Calc Flux**********/
  float* global_max;
  blocksPerGrid =						\
    ((X+1)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;

  cudaMalloc(&global_max, blocksPerGrid*sizeof(float));
  h_FluxMidX<<<blocksPerGrid, threadsPerBlock>>>(F_mid, global_max, FLX, FRX, UL,UR, physL, physR, (X+1)*(Y)*(Z));
  /*****************************/
  /*********Free Cuda Memory*******/
  cudaFree(physL);
  cudaFree(physR);
  cudaFree(FLX);
  cudaFree(FRX);
  cudaFree(UL);
  cudaFree(UR);

  
  /*****************************/
  blocksPerGrid =						\
    ((X+1)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  float* h_global_max;
  h_global_max = (float *) malloc(blocksPerGrid*sizeof(float));
  cudaMemcpy(h_global_max, global_max, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(global_max);
  float temp=0;
  for(int i = 0; i < blocksPerGrid; i++)
    {
      if(temp < h_global_max[i]) temp = h_global_max[i];
    }
  *max = temp;
  
  /*
  int blocksPerGrid1;
  while( blocksPerGrid > BLOCK_SIZE ){
    blocksPerGrid1 = (blocksPerGrid + threadsPerBlock - 1)/threadsPerBlock;
    float *out;
    cudaMalloc(&out, blocksPerGrid1*sizeof(float));
    h_Max<<<blocksPerGrid1, threadsPerBlock>>>(out, global_max, blocksPerGrid);
    Check_CUDA_Error("at RiemansolverX: h_Max");
    cudaFree(global_max);
    cudaMalloc(&global_max, blocksPerGrid1*sizeof(float));
    cudaMemcpy(global_max, out, blocksPerGrid1*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(out);
    blocksPerGrid = blocksPerGrid1;
  }
  blocksPerGrid1 = (blocksPerGrid + threadsPerBlock - 1)/threadsPerBlock;
  float *out;
  cudaMalloc(&out, blocksPerGrid1*sizeof(float));
  h_Max<<<blocksPerGrid1, threadsPerBlock>>>(out, global_max, blocksPerGrid);
  Check_CUDA_Error("at RiemansolverX: h_Max");
  cudaFree(global_max);
  cudaMemcpy(max, out, sizeof(float),cudaMemcpyDeviceToHost);
  Check_CUDA_Error("at RiemansolverX: Mcp global_max ");
  cudaFree(out);  */
}


void riemansolverY(float *F_mid, float *U, float *max)
{
  int threadsPerBlock, blocksPerGrid;
  threadsPerBlock = BLOCK_SIZE;
  /**************Ucalcinv*********/
  float* phys; 
  cudaMalloc(&phys, 8*(X)*(Y)*(Z)*sizeof(float));
  blocksPerGrid =    				\
    ((X)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_Ucalcinv<<<blocksPerGrid, threadsPerBlock>>>(phys,U,(X)*(Y)*(Z));
  /****************************/
  /*************Set_Boundary*********/
  float *phys_temp; 
  cudaMalloc(&phys_temp, 8*(X)*(Y+4)*(Z)*sizeof(float));  
  blocksPerGrid =						\
    ((X)*(Y+4)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_BoundaryY<<<blocksPerGrid, threadsPerBlock>>>(phys_temp, phys, (X)*(Y+4)*(Z));
  /**********************************/

  /**************************/
  
  cudaFree(phys);
  
  /***********Set_physState***************/
  float *physL;
  cudaMalloc(&physL, 8*(X)*(Y+1)*(Z)*sizeof(float));
  float *physR;
  cudaMalloc(&physR, 8*(X)*(Y+1)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y+1)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  
  h_StateY<<<blocksPerGrid, threadsPerBlock>>>(physL, physR, phys_temp,(X)*(Y+1)*(Z));
  /**************************************/

  cudaFree(phys_temp);

  /***********************************/
  float *FLY;
  cudaMalloc(&FLY, 8*(X)*(Y+1)*(Z)*sizeof(float));
  float *FRY;
  cudaMalloc(&FRY, 8*(X)*(Y+1)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y+1)*(Z) + threadsPerBlock - 1) / threadsPerBlock;  
  h_FluxCalcPY_N<<<blocksPerGrid, threadsPerBlock>>>(FLY,  physL, (X)*(Y+1)*(Z));
  h_FluxCalcPY_N<<<blocksPerGrid, threadsPerBlock>>>(FRY,  physR, (X)*(Y+1)*(Z));

  /********************************/
  
  /****************Calc U**********/
  float* UL;
  cudaMalloc(&UL, 8*(X)*(Y+1)*(Z)*sizeof(float));
  float* UR;
  cudaMalloc(&UR, 8*(X)*(Y+1)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y+1)*(Z) + threadsPerBlock - 1) / threadsPerBlock;  
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UL, physL, (X)*(Y+1)*(Z));
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UR, physR, (X)*(Y+1)*(Z));
  /********************************/

  /***************Calc Flux***********/
  float* global_max;
  cudaMalloc(&global_max, blocksPerGrid*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y+1)*(Z) + threadsPerBlock - 1) / threadsPerBlock;  
  h_FluxMidY<<<blocksPerGrid, threadsPerBlock>>>(F_mid, global_max, FLY, FRY, UL,UR, physL, physR, (X)*(Y+1)*(Z));
  /**********************************/
  cudaFree(physL);
  cudaFree(physR);
  cudaFree(FLY);
  cudaFree(FRY);
  cudaFree(UL);
  cudaFree(UR);
  /***********************************/
  
  blocksPerGrid =						\
    ((X)*(Y+1)*(Z) + threadsPerBlock - 1) / threadsPerBlock;

  float* h_global_max;
  h_global_max = (float *) malloc(blocksPerGrid*sizeof(float));
  cudaMemcpy(h_global_max, global_max, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(global_max);
  float temp=0;
  for(int i = 0; i < blocksPerGrid; i++)
    {
      if(temp < h_global_max[i]) temp = h_global_max[i];
    }
  *max = temp;
  free(h_global_max);
 }


void riemansolverZ(float *F_mid, float *U, float *max)
{
  //  float* d_r_group;
  //cudaMalloc(&d_r_group, 3*(X)*(Y)*(Z+1)*sizeof(float));
  int threadsPerBlock, blocksPerGrid;
  threadsPerBlock = BLOCK_SIZE;
  /*********Ucalcinv*************/
  float* phys; 
  cudaMalloc(&phys, 8*(X)*(Y)*(Z)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_Ucalcinv<<<blocksPerGrid, threadsPerBlock>>>(phys,U,(X)*(Y)*(Z));

  /******************************/
  /**********Set_Boundary*******/
  float *phys_temp; 
  cudaMalloc(&phys_temp, 8*(X)*(Y)*(Z+4)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y)*(Z+4) + threadsPerBlock - 1) / threadsPerBlock;
  h_BoundaryZ<<<blocksPerGrid, threadsPerBlock>>>(phys_temp, phys, (X)*(Y)*(Z+4));
  /****************************/

  cudaFree(phys);


  /***********Set_physState******/
  float *physL;
  cudaMalloc(&physL, 8*(X)*(Y)*(Z+1)*sizeof(float));
  float *physR;
  cudaMalloc(&physR, 8*(X)*(Y)*(Z+1)*sizeof(float));


  blocksPerGrid =						\
    ((X)*(Y)*(Z+1) + threadsPerBlock - 1) / threadsPerBlock;
  h_StateZ<<<blocksPerGrid, threadsPerBlock>>>(physL, physR, phys_temp,(X)*(Y)*(Z+1));
  //  cudaDeviceSynchronize();
  /******************************/

  cudaFree(phys_temp);



  /**********Calc_Flux***********/
  float *FLZ;
  cudaMalloc(&FLZ, 8*(X)*(Y)*(Z+1)*sizeof(float));
  float *FRZ;
  cudaMalloc(&FRZ, 8*(X)*(Y)*(Z+1)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y)*(Z+1) + threadsPerBlock - 1) / threadsPerBlock;
  
  h_FluxCalcPZ_N<<<blocksPerGrid, threadsPerBlock>>>(FLZ,  physL, (X)*(Y)*(Z+1));
  h_FluxCalcPZ_N<<<blocksPerGrid, threadsPerBlock>>>(FRZ,  physR, (X)*(Y)*(Z+1));

  /*****************************/

  /***********Calc U************/
  float* UL;
  cudaMalloc(&UL, 8*(X)*(Y)*(Z+1)*sizeof(float));
  float* UR;
  cudaMalloc(&UR, 8*(X)*(Y)*(Z+1)*sizeof(float));
  blocksPerGrid =						\
    ((X)*(Y)*(Z+1) + threadsPerBlock - 1) / threadsPerBlock;
  
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UL, physL, (X)*(Y)*(Z+1));
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(UR, physR, (X)*(Y)*(Z+1));

  /*********Calc Flux**************/
  float* global_max;
  blocksPerGrid =						\
    ((X)*(Y)*(Z+1) + threadsPerBlock - 1) / threadsPerBlock;

  cudaMalloc(&global_max, blocksPerGrid*sizeof(float));
  h_FluxMidZ<<<blocksPerGrid, threadsPerBlock>>>(F_mid, global_max, FLZ, FRZ, UL,UR, physL, physR, (X)*(Y)*(Z+1));
  /*********************************/

  /**********Free Cuda Memory********/

  cudaFree(physL);
  cudaFree(physR);
  cudaFree(FLZ);
  cudaFree(FRZ);
  cudaFree(UL);
  cudaFree(UR);
  /********************************/
  
  blocksPerGrid =						\
    ((X)*(Y)*(Z+1) + threadsPerBlock - 1) / threadsPerBlock;

  float* h_global_max;
  h_global_max = (float *) malloc(blocksPerGrid*sizeof(float));
  cudaMemcpy(h_global_max, global_max, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(global_max);
  float temp=0;
  for(int i = 0; i < blocksPerGrid; i++)
    {
      if(temp < h_global_max[i]) temp = h_global_max[i];
    }
  *max = temp;
  free(h_global_max);
}

__global__ void h_Set_Potential(float *Potential, float dx, float dy, float dz, int NThreads)
{
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  int i,j,k,N;
  //  float x, y;
  float z;
  int Sx = (Y+2)*(Z+2), Sy = (Z+2), Sz = 1;
  
  if( threadID < NThreads){
    i = threadID / (Y+2)*(Z+2);
    j = threadID /(Z+2) - i*(Y+2);
    k = threadID - (threadID/(Z+2))*(Z+2);
    //   x = XMIN + dx * (i - 1);
    //y = YMIN + dy * (j - 1);
    z = ZMIN + dz * (k - 1);
    N = (i*Sx + j*Sy + k*Sz);
    Potential[N] = G * z;
  }
}

__global__ void h_Cal_temp(float *phys, float *temp, float dx, float dy, float dz, int NThreads) 
{
  int i,j,k,N,Nf;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int Sx = (Y)*(Z), Sy = (Z), Sz = 1;
  float ee = 6.02e23 * 1.6e-19 * 1.e3;

  if(threadID < NThreads){
    i = threadID/(Y)*(Z);
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    N = i*Sx*3+j*Sy*3+k*3;
    Nf = i*Sx*8+j*Sy*8+k*8;
    if(k>0&&k<Z-1) {
      temp[N]   = GradientX(phys, Nf, 8, 4, dx) / (phys[Nf]*ee);
      temp[N+1] = GradientY(phys, Nf, 8, 4, dy) / (phys[Nf]*ee);
      temp[N+2] = GradientZ(phys, Nf, 8, 4, dz) / (phys[Nf]*ee);
    }
    else {
      temp[N] = 0.0;
      temp[N+1] = 0.0;
      temp[N+2] = 0.0;
    }
  }
}

__global__ void h_Cal_temp2(float *temp, float *temp2, float dx, float dy, float dz, int NThreads) 
{
  int i,j,k,N;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int Sx = (Y)*(Z), Sy = (Z), Sz = 1;
  
  if(threadID < NThreads){
    i = threadID/(Y)*(Z);
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    N = i*Sx*3+j*Sy*3+k*3;
    if(k>0&&k<Z-1) {
      temp2[N]   = GradientY(temp, N, 3, 2, dy)-GradientZ(temp, N, 3, 1, dz);
      temp2[N+1] = GradientZ(temp, N, 3, 0, dz)-GradientX(temp, N, 3, 2, dx);
      temp2[N+2] = GradientX(temp, N, 3, 1, dx)-GradientY(temp, N, 3, 0, dy);
    }
    else {
      temp2[N] = 0.0;
      temp2[N+1] = 0.0;
      temp2[N+2] = 0.0;
    }
  }
}
__global__ void h_Cal_Source(float *Source, float *phys, float *temp2, float GridRatioX, \
			float GridRatioY, float GridRatioZ, float dt, int NThreads)
{
  int i,j,k,N,Nf;
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int Sx = (Y)*(Z), Sy = (Z), Sz = 1;
  float gx, gy, gz;
  
  if(threadID < NThreads){
    i = threadID/(Y)*(Z);
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    gx = 0;//(Potential[N + Sx2] - Potential[N - Sx2]) * GridRatioX / 2;
    gy = 0;//(Potential[N + Sy2] - Potential[N - Sy2]) * GridRatioY / 2;
    gz = G * dt;//(Potential[N + Sz2] - Potential[N - Sz2]) * GridRatioZ / 2;
    Nf = 8*(i*Sx + j*Sy + k*Sz);
    N = 3*(i*Sx + j*Sy + k*Sz);
    Source[Nf+0] = 0.;
    Source[Nf+1] = - gx * phys[Nf+0];
    Source[Nf+2] = - gy * phys[Nf+0];
    Source[Nf+3] = - gz * phys[Nf+0];
    Source[Nf+4] = - phys[Nf+0]*(gx * phys[Nf+1] + gy * phys[Nf+2]\
				 + gz * phys[Nf+3]);
    Source[Nf+5] = temp2[N]   * dt;
    Source[Nf+6] = temp2[N+1] * dt;
    Source[Nf+7] = temp2[N+2] * dt;
  }
}
    
void h_Fluxsource( float *Source, float *U, float *dt)
{
  float *phys;
  float *temp, *temp2;
  cudaMalloc(&phys, 8*(X)*(Y)*(Z)*sizeof(float));
  cudaMalloc(&temp, 3*(X)*(Y)*(Z)*sizeof(float));
  cudaMalloc(&temp2, 3*(X)*(Y)*(Z)*sizeof(float));

  float dx, dy, dz;

  Grid(&dx, &dy, &dz);

  float GridRatioX = *dt/dx;
  float GridRatioY = *dt/dy;
  float GridRatioZ = *dt/dz;
  
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = \
    ( (X)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock;
  h_Ucalcinv<<<blocksPerGrid, threadsPerBlock>>>(phys, U, (X)*(Y)*(Z));
#if 0
  float *Potential;
  cudaMalloc(&Potential, (X+2)*(Y+2)*(Z+2)*sizeof(float));
  Check_CUDA_Error("at Fluxsource: Malloc Potential");
  blocksPerGrid = \
    ((X+2)*(Y+2)*(Z+2) + threadsPerBlock - 1)/ threadsPerBlock;
  h_Set_Potential<<<blocksPerGrid, threadsPerBlock>>>(Potential, dx, dy, dz, (X+2)*(Y+2)*(Z+2));
  Check_CUDA_Error("at Fluxsource: Set_Potential");
#endif 
    
  blocksPerGrid = \
    ((X)*(Y)*(Z) + threadsPerBlock - 1) / threadsPerBlock; 
  h_Cal_temp<<<blocksPerGrid, threadsPerBlock>>>(phys, temp, dx, dy, dz, (X)*(Y)*(Z));
  h_Cal_temp2<<<blocksPerGrid, threadsPerBlock>>>(temp, temp2, dx, dy, dz, (X)*(Y)*(Z));
  h_Cal_Source<<<blocksPerGrid, threadsPerBlock>>>(Source, phys, temp2, GridRatioX, GridRatioY, GridRatioZ, *dt, (X)*(Y)*(Z));
  cudaFree(phys);
  cudaFree(temp);
  cudaFree(temp2);
  //cudaFree(Potential);
  //Check_CUDA_Error("at Fluxsource: cudaFree Potential");

}
  
__global__ void h_U_update1(float *U1, float *U_old,\
			    float *FmidX, float *FmidY, float *FmidZ,\
			    float GridRatioX, float GridRatioY, float GridRatioZ,\
			    int NThreads)
{
  int i,j,k,N,Nx,Ny,Nz;
  int Sx  = 8*Y*Z,     Sy = 8*Z,      Sz = 8;
  int Sxx = 8*Y*Z,     Syx = 8*Z,     Szx = 8;
  int Sxy = 8*(Y+1)*Z, Syy = 8*Z,     Szy = 8;
  int Sxz = 8*Y*(Z+1), Syz = 8*(Z+1), Szz = 8;
  
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float d_FmidX[BLOCK_SIZE*2];
  __shared__ float d_FmidY[BLOCK_SIZE*2];
  __shared__ float d_FmidZ[BLOCK_SIZE*2];
  __shared__ float d_U_old[BLOCK_SIZE];
  
  if(threadID < NThreads){
    i = threadID/((Y)*(Z));
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    for(int l=0; l<8; l++){
      N  = i*Sx + j*Sy + k*Sz + l;
      Nx = i*Sxx + j*Syx + k*Szx + l;
      Ny = i*Sxy + j*Syy + k*Szy + l;
      Nz = i*Sxz + j*Syz + k*Szz + l;
      d_FmidX[2*threadIdx.x + 0] = FmidX[Nx];
      d_FmidX[2*threadIdx.x + 1] = FmidX[Nx+Sxx];
      d_FmidY[2*threadIdx.x + 0] = FmidY[Ny];
      d_FmidY[2*threadIdx.x + 1] = FmidY[Ny+Syy];
      d_FmidZ[2*threadIdx.x + 0] = FmidZ[Nz];
      d_FmidZ[2*threadIdx.x + 1] = FmidZ[Nz+Szz];
      d_U_old[threadIdx.x] = U_old[N];
      
      U1[N] = d_U_old[threadIdx.x] + (- GridRatioX * (d_FmidX[2*threadIdx.x+1] - d_FmidX[2*threadIdx.x+0]) - GridRatioY * (d_FmidY[2*threadIdx.x+1] - d_FmidY[2*threadIdx.x+0]) - GridRatioZ * (d_FmidZ[2*threadIdx.x+1] - d_FmidZ[2*threadIdx.x+0]));
      __syncthreads();
    }
    
  }
}

  
__global__ void h_U_update2(float *U2, float *U1, float *U_old, \
			    float *FmidX, float *FmidY, float *FmidZ,\
			    float GridRatioX, float GridRatioY, float GridRatioZ,\
			    int NThreads)
{
  int i,j,k,N,Nx,Ny,Nz;
  int Sx  = 8*Y*Z,     Sy = 8*Z,      Sz = 8;
  int Sxx = 8*Y*Z,     Syx = 8*Z,     Szx = 8;
  int Sxy = 8*(Y+1)*Z, Syy = 8*Z,     Szy = 8;
  int Sxz = 8*Y*(Z+1), Syz = 8*(Z+1), Szz = 8;
  
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float d_FmidX[BLOCK_SIZE*2];
  __shared__ float d_FmidY[BLOCK_SIZE*2];
  __shared__ float d_FmidZ[BLOCK_SIZE*2];
  __shared__ float d_U_old[BLOCK_SIZE];
  __shared__ float d_U1[BLOCK_SIZE];
  
  if(threadID < NThreads){
    i = threadID/((Y)*(Z));
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    for(int l=0; l<8; l++){
      N  = i*Sx + j*Sy + k*Sz + l;

      Nx = i*Sxx + j*Syx + k*Szx + l;
      Ny = i*Sxy + j*Syy + k*Szy + l;
      Nz = i*Sxz + j*Syz + k*Szz + l;

      d_FmidX[2*threadIdx.x + 0] = FmidX[Nx];
      d_FmidX[2*threadIdx.x + 1] = FmidX[Nx+Sxx];
      d_FmidY[2*threadIdx.x + 0] = FmidY[Ny];
      d_FmidY[2*threadIdx.x + 1] = FmidY[Ny+Syy];
      d_FmidZ[2*threadIdx.x + 0] = FmidZ[Nz];
      d_FmidZ[2*threadIdx.x + 1] = FmidZ[Nz+Szz];
      d_U_old[threadIdx.x] = U_old[N];
      d_U1[threadIdx.x] = U1[N];
 
      U2[N] = 0.75*d_U_old[threadIdx.x] + 0.25 * d_U1[threadIdx.x] + 0.25*(- GridRatioX * (d_FmidX[2*threadIdx.x+1] - d_FmidX[2*threadIdx.x+0]) - GridRatioY * (d_FmidY[2*threadIdx.x+1] - d_FmidY[2*threadIdx.x+0]) - GridRatioZ * (d_FmidZ[2*threadIdx.x+1] - d_FmidZ[2*threadIdx.x+0]));
      __syncthreads();
    }
  }
}

  
__global__ void h_U_update3(float *U_new, float *U2, float *U_old,float *Source, \
			    float *FmidX, float *FmidY, float *FmidZ,\
			    float GridRatioX, float GridRatioY, float GridRatioZ,\
			    int NThreads)
{
  int i,j,k,N,Nx,Ny,Nz;
  int Sx  = 8*Y*Z,     Sy = 8*Z,      Sz = 8;
  int Sxx = 8*Y*Z,     Syx = 8*Z,     Szx = 8;
  int Sxy = 8*(Y+1)*Z, Syy = 8*Z,     Szy = 8;
  int Sxz = 8*Y*(Z+1), Syz = 8*(Z+1), Szz = 8;
  
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadID < NThreads){
    i = threadID/((Y)*(Z));
    j = threadID/(Z) - i*(Y);
    k = threadID - (threadID/(Z))*(Z);
    for(int l=0; l<8; l++){
      N  = i*Sx + j*Sy + k*Sz + l;

      Nx = i*Sxx + j*Syx + k*Szx + l;
      Ny = i*Sxy + j*Syy + k*Szy + l;
      Nz = i*Sxz + j*Syz + k*Szz + l;
      U_new[N] = (1./3.)*U_old[N] + (2./3.) * U2[N] + (2./3.)*(- GridRatioX * (FmidX[Nx+Sxx] - FmidX[Nx]) - GridRatioY * (FmidY[Ny+Syy] - FmidY[Ny]) - GridRatioZ * (FmidZ[Nz+Szz] - FmidZ[Nz])) + Source[N];
    }
  }
}


void Advance(float *U_new, float *U_old, float *dt){


  float h_maxX=0;
  float h_maxY=0;
  float h_maxZ=0;
  float max;
  
  float dx, dy, dz;
  Grid(&dx, &dy, &dz);
  float GridRatioX = *dt/dx;
  float GridRatioY = *dt/dy;
  float GridRatioZ = *dt/dz;

  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid;

  //timestamp_type time1, time2;


  /*******riemansolverX***********/
  //  get_timestamp(&time1);
  float *d_FmidX;
  cudaMalloc((void **)&d_FmidX, 8*(X+1)*(Y)*(Z)*sizeof(float));
  riemansolverX(d_FmidX, U_old, &max);
  //cudaDeviceSynchronize();
  //get_timestamp(&time2);
  //printf("time for riemansolverX: %f s\n", timestamp_diff_in_seconds(time1,time2));
  if( max > h_maxX) h_maxX = max;

  /*  float*h_FmidX;
  cudaMallocHost(&h_FmidX, 5*(X+1)*(Y)*(Z)*sizeof(float));
  cudaMemcpy(h_FmidX, d_FmidX, 5*(X+1)*(Y)*(Z)*sizeof(float), cudaMemcpyDeviceToHost);
  output_print_all(h_FmidX,"h_FmidX", X+1, Y, Z, (X+1)*(Y)*(Z), 5);
  cudaFreeHost(h_FmidX);
  printf("RiemanX Max: %f\n", max);
  */
  
  /*******riemansolverZ*********/
  float *d_FmidZ;
  cudaMalloc((void **)&d_FmidZ, 8*(X)*(Y)*(Z+1)*sizeof(float));
  riemansolverZ(d_FmidZ, U_old, &max);
  if( max > h_maxZ ) h_maxZ = max;

  /*******riemansolverY**********/
  float *d_FmidY;
  cudaMalloc((void **)&d_FmidY, 8*(X)*(Y+1)*(Z)*sizeof(float));
  riemansolverY(d_FmidY, U_old, &max);
  if( max > h_maxY ) h_maxY = max;
  
 
  /******Fluxsource**************/
  float *d_Source;
  cudaMalloc((void **)&d_Source, 8*(X)*(Y)*(Z)*sizeof(float));
  h_Fluxsource(d_Source, U_old, dt);
  cudaDeviceSynchronize();
  /***********U_update************/
  float *d_U1;
  cudaMalloc((void **)&d_U1, 8*(X)*(Y)*(Z)*sizeof(float));  
  blocksPerGrid = \
    ( (X)*(Y)*(Z) + threadsPerBlock - 1)/threadsPerBlock;
  h_U_update1<<<blocksPerGrid, threadsPerBlock>>>(d_U1, U_old, \
						  d_FmidX, d_FmidY, d_FmidZ,\
						  GridRatioX, GridRatioY, GridRatioZ,\
						  (X)*(Y)*(Z));
  
  riemansolverX(d_FmidX, d_U1, &max);
  if( max > h_maxX) h_maxX = max;
  riemansolverY(d_FmidY, d_U1, &max);
  if( max > h_maxY ) h_maxY = max;
  riemansolverZ(d_FmidZ, d_U1, &max);
  if( max > h_maxZ ) h_maxZ = max;
  h_Fluxsource(d_Source, d_U1, dt);


  float *d_U2;
  cudaMalloc((void **)&d_U2, 8*(X)*(Y)*(Z)*sizeof(float));
  blocksPerGrid = \
    ( (X)*(Y)*(Z) + threadsPerBlock - 1)/threadsPerBlock;
  h_U_update2<<<blocksPerGrid, threadsPerBlock>>>(d_U2, d_U1, U_old, \
						  d_FmidX, d_FmidY, d_FmidZ,\
						  GridRatioX, GridRatioY, GridRatioZ,\
						  (X)*(Y)*(Z));


  riemansolverX(d_FmidX, d_U2, &max);
  if( max > h_maxX) h_maxX = max;
  riemansolverY(d_FmidY, d_U2, &max);
  if( max > h_maxY ) h_maxY = max;
  riemansolverZ(d_FmidZ, d_U2, &max);
  if( max > h_maxZ ) h_maxZ = max;

  h_Fluxsource(d_Source, d_U2, dt);

  blocksPerGrid = \
    ( (X)*(Y)*(Z) + threadsPerBlock - 1)/threadsPerBlock;
  h_U_update3<<<blocksPerGrid, threadsPerBlock>>>(U_new, d_U2, U_old, d_Source,	\
						  d_FmidX, d_FmidY, d_FmidZ,\
						  GridRatioX, GridRatioY, GridRatioZ,\
						  (X)*(Y)*(Z));

  Grid(&GridRatioX, &GridRatioY, &GridRatioZ);
  float dtX = 0.3 * GridRatioX / h_maxX;
  float dtY = 0.3 * GridRatioY / h_maxY;
  float dtZ = 0.3 * GridRatioZ / h_maxZ;

  
  if (dtY<dtX) *dt = dtY;
  else *dt = dtX;
  if (dtZ<*dt) *dt = dtZ;

  cudaFree(d_FmidX);
  cudaFree(d_FmidY);
  cudaFree(d_FmidZ);
  cudaFree(d_U1);
  cudaFree(d_U2);
  cudaFree(d_Source);
}


int main()
{
  int NCell = (X)*(Y)*(Z);
  size_t size;
  float *h_U = (float *) malloc(8*(X)*(Y)*(Z)*sizeof(float));
  float dt = 1.e-13;
  float t = 0 ;
  int k=0,l=0;
  
  Creat_folder();//Creat folder to store data;
  /*****************/
  size = 8*(X)*(Y)*(Z)*sizeof(float);
  float *d_phys;
  cudaMalloc((void **)&d_phys,size );
  float *d_U;
  cudaMalloc((void **)&d_U,size);
  float *d_U_adv;
  cudaMalloc((void **)&d_U_adv,size);
  // launch NCell = (X)*(Y)*(Z) threads
  
  int threadsPerBlock = BLOCK_SIZE;
  int blocksPerGrid = \
    (NCell + threadsPerBlock - 1) / threadsPerBlock;

  
  init_conditions<<<blocksPerGrid, threadsPerBlock>>>(d_phys, NCell);
  Check_CUDA_Error("init_conditions");
  h_Ucalc<<<blocksPerGrid, threadsPerBlock>>>(d_U, d_phys, NCell);
  cudaFree(d_phys);

  
  /*****************Advance*************/
#if 0
  timestamp_type time1, time2;
  get_timestamp(&time1);
  Advance(d_U_adv, d_U, &dt);
  get_timestamp(&time2);
  printf("time for Advance: %f s\n", timestamp_diff_in_seconds(time1,time2));
#endif 
#if 1
  while(t < tmax){
    t+=dt;
    Advance(d_U_adv, d_U, &dt);
    cudaMemcpy(d_U, d_U_adv, 8*(X)*(Y)*(Z)*sizeof(float), cudaMemcpyDeviceToDevice);
    if( t/1.e-9 >0.5 * k){
      cudaMemcpy(h_U, d_U, 8*(X)*(Y)*(Z)*sizeof(float), cudaMemcpyDeviceToHost);
      output_file3(h_U, t/1.e-9);
      //output_file2(h_U, t/1.e-9);
      k++;
    }
    printf ("%d| t = %f\n",l,t/1.e-9);
    l++;
  }
  //cudaMemcpy(h_U, d_U, 8*(X)*(Y)*(Z)*sizeof(float), cudaMemcpyDeviceToHost);
  //output_file3(h_U, t/1.e-9);
  //output_file2(h_U, t/1.e-9);
#endif
  cudaFree(d_U);
  cudaFree(d_U_adv);
  free(h_U);
  //  cudaDeviceReset();
  //cudaStreamDestroy(0);
}
  
   
  
