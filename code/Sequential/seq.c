#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "timing.h"

#define PI    3.14159265
#define G     .1
#define GAMMA 1.4
#define X 200
#define Y 2
#define Z 200
#define XMIN -.25
#define XMAX .25
#define YMIN -.25
#define YMAX .25
#define ZMIN -.75
#define ZMAX .75
#define THETA 2.0
#define tmax 12

void Grid(double *gridX, double *gridY, double *gridZ) {
  *gridX = (XMAX - XMIN) / (X - 1);
  *gridY = (YMAX - YMIN) / (Y - 1);
  *gridZ = (ZMAX - ZMIN) / (Z - 1);
}

double sgn(double x) {					// Gives the sign of x
  double S;
  if (x>0.) S = 1.;
  else S = -1.;
  return S;
}

double minmod(double x, double y, double z) {		// The minmod function, described in the paper
  double min, M;
  min = fabs(x);
  if (fabs(y) < min) min = fabs(y);
  if (fabs(z) < min) min = fabs(z);
  M = .25 * fabs(sgn(x) + sgn(y)) * (sgn(x) + sgn(z)) * min;
  return M;
}

double GradientX(double *phys, int N, int L, int O, double dx) {
  if (N < L*Y*Z) {
    return (phys[N+L*Y*Z+O]-phys[N+(X-1)*L*Y*Z+O])/dx;
  }
  else if (N >= (X-1)*L*Y*Z) {
    return (phys[N-(X-1)*L*Y*Z+O]-phys[N-L*Y*Z+O])/dx;
  }
  else {
    return (phys[N+L*Y*Z+O]-phys[N-L*Y*Z+O])/dx;
  }
}

double GradientY(double *phys, int N, int L, int O, double dy) {
  if (N%(L*Y*Z) < L*Z) {
    return (phys[N+L*Z+O]-phys[N+(Y-1)*L*Z+O])/dy;
  }
  else if (N%(L*Y*Z) >= (Y-1)*L*Z) {
    return (phys[N-(Y-1)*L*Z+O]-phys[N-L*Z+O])/dy;
  }
  else {
    return (phys[N+L*Z+O]-phys[N-L*Z+O])/dy;
  }
}

double GradientZ(double *phys, int N, int L, int O, double dz) {
  return (phys[N+L+O]-phys[N-L+O])/dz;
}

// Given the density, speed and pressure, we calculate the flux F:
void FluxCalcPX(double *FX, double *phys, int N) {
  int i=0;
  for (i=0;i<N;i++) {
      FX[5*i+0] = phys[5*i+0] * phys[5*i+1];
      FX[5*i+1] = phys[5*i+0] * phys[5*i+1] * phys[5*i+1] + phys[5*i+4];
      FX[5*i+2] = phys[5*i+0] * phys[5*i+1] * phys[5*i+2];
      FX[5*i+3] = phys[5*i+0] * phys[5*i+1] * phys[5*i+3];
      FX[5*i+4] = phys[5*i+1] *(phys[5*i+0] *(phys[5*i+1] * phys[5*i+1] + phys[5*i+2] * phys[5*i+2] + phys[5*i+3] * phys[5*i+3]) * .5 + phys[5*i+4] * GAMMA / (GAMMA - 1));

  }
}

void FluxCalcPY(double *FY, double *phys, int N) {
  int i=0;
  for (i=0;i<N;i++) {
      
      FY[5*i+0] = phys[5*i+0] * phys[5*i+2];
      FY[5*i+1] = phys[5*i+0] * phys[5*i+2] * phys[5*i+1];
      FY[5*i+2] = phys[5*i+0] * phys[5*i+2] * phys[5*i+2] + phys[5*i+4];
      FY[5*i+3] = phys[5*i+0] * phys[5*i+2] * phys[5*i+3];
      FY[5*i+4] = phys[5*i+2] *(phys[5*i+0] *(phys[5*i+1] * phys[5*i+1] + phys[5*i+2] * phys[5*i+2] + phys[5*i+3] * phys[5*i+3]) * .5 + phys[5*i+4] * GAMMA / (GAMMA - 1));

  }
}

void FluxCalcPZ(double *FZ, double *phys, int N) {
  int i=0;
  for (i=0;i<N;i++) {
      
      FZ[5*i+0] = phys[5*i+0] * phys[5*i+3];
      FZ[5*i+1] = phys[5*i+0] * phys[5*i+3] * phys[5*i+1];
      FZ[5*i+2] = phys[5*i+0] * phys[5*i+3] * phys[5*i+2];
      FZ[5*i+3] = phys[5*i+0] * phys[5*i+3] * phys[5*i+3] + phys[5*i+4];
      FZ[5*i+4] = phys[5*i+3] *(phys[5*i+0] *(phys[5*i+1] * phys[5*i+1] + phys[5*i+2] * phys[5*i+2] + phys[5*i+3] * phys[5*i+3]) * .5 + phys[5*i+4] * GAMMA / (GAMMA - 1));

  }
}

// Given the density, speed and pressure, we calculate the conserved variable U:
void Ucalc(double *U, double *physicalvar, int N) {     // physicalvar[] = rho, vx, vy, vz, p
  int i=0;
  for (i=0;i<N;i++) {
    U[5*i+0] = physicalvar[5*i+0];
    U[5*i+1] = physicalvar[5*i+0] * physicalvar[5*i+1];
    U[5*i+2] = physicalvar[5*i+0] * physicalvar[5*i+2];
    U[5*i+3] = physicalvar[5*i+0] * physicalvar[5*i+3];
    U[5*i+4] = physicalvar[5*i+4] / (GAMMA - 1) + 0.5 * physicalvar[5*i+0] * (physicalvar[5*i+1] * physicalvar[5*i+1] + physicalvar[5*i+2] * physicalvar[5*i+2] + physicalvar[5*i+3] * physicalvar[5*i+3]);
  }
}

// And the inverse:
void Ucalcinv(double *physicalvar, double *U, int N) {     // physicalvar[] = rho, v, p
  int i=0;
  for (i=0;i<N;i++) {
    physicalvar[5*i+0] = U[5*i+0];
    physicalvar[5*i+1] = U[5*i+1] / U[5*i+0];
    physicalvar[5*i+2] = U[5*i+2] / U[5*i+0];
    physicalvar[5*i+3] = U[5*i+3] / U[5*i+0];
    physicalvar[5*i+4] = (U[5*i+4] - 0.5 * (U[5*i+1] * U[5*i+1] + U[5*i+2] * U[5*i+2] + U[5*i+3] * U[5*i+3])/ U[5*i+0]) * (GAMMA - 1);
  }
}


void init(double *physical) {
  int i=0, j=0, k=0, N;
  double   dx, dy, dz;
  Grid (&dx, &dy, &dz);
  double x, y, z;

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        x = XMIN + dx * i;
        y = YMIN + dy * j;
        z = ZMIN + dz * k;
        N = 5 * (i*Y*Z + j*Z + k);
        if (x*x+z*z<0.1) {
          physical[N+0] = 1.0;
          physical[N+1] = 0.0;
          physical[N+2] = 0.0;
          physical[N+3] = 0.0;
          physical[N+4] = 1.0;
        }
        else {
          physical[N+0] = 0.1;
          physical[N+1] = 0.0;
          physical[N+2] = 0.0;
          physical[N+3] = 0.0;
          physical[N+4] = 0.125;
        }
      }
    }
  }
}

void init_water_oil(double *physical) {
  int i=0, j=0, k=0, N;
  double   dx, dy, dz, P0=2.5, rhol=1., rhoh=2.;
  Grid (&dx, &dy, &dz);
  double x, y, z;

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        x = XMIN + dx * i;
        y = YMIN + dy * j;
        z = ZMIN + dz * k;
        N = 5	 * (i*Y*Z + j*Z + k);

        if (z<0.0*cos(x*6*3.1415)) {
          physical[N+0] = rhol;
          physical[N+1] = 0.0;
          physical[N+2] = 0.0;
          physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
          physical[N+4] = P0-rhol*G*(z-ZMIN);
        }
        else {
          physical[N+0] = rhoh;
          physical[N+1] = 0.0;
          physical[N+2] = 0.0;
          physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
          physical[N+4] = P0-rhol*G*(-ZMIN)-rhoh*G*(z);
        }
      }
    }
  }
}

void init_water_oil_3d(double *physical) {
    int i=0, j=0, k=0, N;
    double   dx, dy, dz, P0=2.5, rhol=1., rhoh=2.;
    Grid (&dx, &dy, &dz);
    double x, y, z;
    
    for (i=0;i<X;i++) {
        for (j=0;j<Y;j++) {
            for (k=0;k<Z;k++) {
                x = XMIN + dx * i;
                y = YMIN + dy * j;
                z = ZMIN + dz * k;
                N = 5	 * (i*Y*Z + j*Z + k);
                
                if (z < 0.0*cos(x*6*3.1415)) {
                    physical[N+0] = rhol;
                    physical[N+1] = 0.0;
                    physical[N+2] = 0.0;
                    physical[N+3] = 0.01*(1. + cos(2*PI*x/((XMAX)-(XMIN))))*(1. + cos(2*PI*y/((YMAX)-(YMIN))))*(1.+cos(2*PI*z/((ZMAX)-(ZMIN))))/8.;
                    
                    //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*y))*(1. + cos(4*PI*z/3.))/8.;
                    //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
                    physical[N+4] = P0-rhol*G*(z-ZMIN);
                } else {
                    physical[N+0] = rhoh;
                    physical[N+1] = 0.0;
                    physical[N+2] = 0.0;
                    physical[N+3] = 0.01*(1. + cos(2*PI*x/((XMAX)-(XMIN))))*(1. + cos(2*PI*y/((YMAX)-(YMIN))))*(1.+cos(2*PI*z/((ZMAX)-(ZMIN))))/8.;
                    //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*y))*(1. + cos(4*PI*z/3.))/8.;
                    //physical[N+3] = 0.01*(1. + cos(4*PI*x))*(1. + cos(4*PI*z/3.))/4.;
                    physical[N+4] = P0-rhol*G*(-ZMIN)-rhoh*G*(z);
                }
            }
        }
    }
}

void init_B_field(double *B) {
    int i=0, j=0, k=0, N;
    for (i=0;i<X;i++) {
        for (j=0;j<Y;j++) {
            for (k=0;k<Z;k++) {
                N = 3	 * (i*Y*Z + j*Z + k);
                
                B[N] = 0.0;
                B[N+1] = 0.0;
                B[N+2] = 0.0;
            }
        }
    }
}

void AdvanceB(double *B, double *phys, double dt) {
  double *E = (double *) malloc (3*X*Y*Z*sizeof(double));
  double dx, dy, dz;
  double mu = 4*3.1415926536e-7, ee = 6.02e23 * 1.6e-19;
  int i, j, k, N1, N2;
  Grid (&dx, &dy, &dz);
  
  for (i = 0; i < X; i++) {
    for (j = 0; j < Y; j++) {
      for (k = 1; k < Z - 1; k++) {
        N1 = i * 3 * Y * Z + j * 3 * Z + k * 3;
        N2 = i * 5 * Y * Z + j * 5 * Z + k * 5;
        E[N1] = (GradientX(phys,N2,5,4,dx)-(B[N1+2]*GradientZ(B,N1,3,0,dz)-B[N1+2]*GradientX(B,N1,3,2,dx)-B[N1+1]*GradientX(B,N1,3,1,dx)+B[N1+1]*GradientY(B,N1,3,0,dy))/mu)/(phys[N2]*ee)+phys[N2+2]*B[N1+2]-phys[N2+3]*B[N1+1];
        E[N1+1] = (GradientY(phys,N2,5,4,dy)-(B[N1]*GradientX(B,N1,3,1,dx)-B[N1]*GradientY(B,N1,3,0,dy)-B[N1+2]*GradientY(B,N1,3,2,dy)+B[N1+2]*GradientZ(B,N1,3,1,dz))/mu)/(phys[N2]*ee)+phys[N2+3]*B[N1]-phys[N2+1]*B[N1+2];
        E[N1+2] = (GradientZ(phys,N2,5,4,dz)-(B[N1+1]*GradientY(B,N1,3,2,dy)-B[N1+1]*GradientZ(B,N1,3,1,dz)-B[N1]*GradientZ(B,N1,3,0,dz)+B[N1]*GradientX(B,N1,3,2,dx))/mu)/(phys[N2]*ee)+phys[N2+1]*B[N1+1]-phys[N2+2]*B[N1];
      }
    }
  }
  for (i = 0; i < X; i++) {
    for (j = 0; j < Y; j++) {
      { 
        k = 0;
        N1 = i * 3 * Y * Z + j * 3 * Z + k * 3;
        E[N1] = 0.0;
        E[N1+1] = 0.0;
        E[N1+2] = 0.0;
      }
      { 
        k = Z-1;
        N1 = i * 3 * Y * Z + j * 3 * Z + k * 3;
        E[N1] = 0.0;
        E[N1+1] = 0.0;
        E[N1+2] = 0.0;
      }
    }
  }

  for (i = 0; i < X; i++) {
    for (j = 0; j < Y; j++) {
      for (k = 1; k < Z - 1; k++) {
        N1 = i * 3 * Y * Z + j * 3 * Z + k * 3;
        B[N1] = B[N1] + dt * (GradientY(E,N1,3,2,dy) - GradientZ(E,N1,3,1,dz));
        B[N1+1] = B[N1+1] + dt * (GradientZ(E,N1,3,0,dz) - GradientX(E,N1,3,2,dx));
        B[N1+2] = B[N1+2] + dt * (GradientX(E,N1,3,1,dx) - GradientY(E,N1,3,0,dy));
      }
    }
  }
  
  free(E);
}

// Calculates the flux at a half integer coordinate using the riemann method.
void riemansolverX(double *F_mid, double *U, double *max) {
  double *phys  = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *phys_temp= (double*) malloc (5*(X+4)*(Y)*(Z)*sizeof (double));
  double *physL = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *physR = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *FLX   = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *FRX   = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *UL    = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *UR    = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double SoundSpeedL, SoundSpeedR;											//L, SoundSpeedR, SoundSpeedLL, SoundSpeedRR;
  double AlphaPlus=0, AlphaMinus=0;
  int i, j, k, l, N;
  int Sx = 5*Y*Z, Sy = 5*Z, Sz = 5;

  Ucalcinv (phys,U,(X)*(Y)*(Z));

  *max=0;

/**********periodic***************/
  for (i=0;i<X+4;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0;l<5;l++) {
          N = Sx*i+Sy*j+Sz*k+l;
          if      (i<2) {
            phys_temp[N] = phys[Sx*(i+X-2)+Sy*j+Sz*k+l];
          }
          else if (i>X+1) {
            phys_temp[N] = phys[Sx*(i-X-2)+Sy*j+Sz*k+l];
          }
          else phys_temp[N] = phys[N-2*Sx];
        }
      }
    }
  }
/************************************/

/***************outflow*****************
  for (i=0;i<X+4;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0;l<5;l++) {
          N = Sx*i+Sy*j+Sz*k+l;
          if      (i<2) {
            phys_temp[N] = phys[Sx*(0)+Sy*j+Sz*k+l];
          }
          else if (i>X+1) {
            phys_temp[N] = phys[Sx*(X-1)+Sy*j+Sz*k+l];
          }
          else phys_temp[N] = phys[N-2*Sx];
        }
      }
    }
  }
/***************************************/

  for (i=0;i<X+1;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0; l<5; l++) {
          N = Sx*i+Sy*j+Sz*k+l;
          physL[N] = phys_temp[N+Sx]   + 0.5 * minmod (THETA*(phys_temp[N+Sx]   - phys_temp[N]),    0.5*(phys_temp[N+2*Sx] - phys_temp[N]),    THETA*(phys_temp[N+2*Sx] - phys_temp[N+Sx]));
          physR[N] = phys_temp[N+2*Sx] - 0.5 * minmod (THETA*(phys_temp[N+2*Sx] - phys_temp[N+Sx]), 0.5*(phys_temp[N+3*Sx] - phys_temp[N+Sx]), THETA*(phys_temp[N+3*Sx] - phys_temp[N+2*Sx]));
        }
      }
    }
  }


  FluxCalcPX (FLX, physL,(X+1)*(Y)*(Z));
  FluxCalcPX (FRX, physR,(X+1)*(Y)*(Z));
  Ucalc     (UL,           physL,(X+1)*(Y)*(Z));
  Ucalc     (UR,           physR,(X+1)*(Y)*(Z));

  for (i=0;i<X+1;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        N = Sx*i+Sy*j+Sz*k;
        AlphaPlus=0.;
        AlphaMinus=0.;
        SoundSpeedL = sqrt (GAMMA * physL[N+4] / physL[N+0]);
        SoundSpeedR = sqrt (GAMMA * physR[N+4] / physR[N+0]);
        if (AlphaPlus  <  physL[N+1] + SoundSpeedL ) AlphaPlus  =  physL[N+1] + SoundSpeedL;
        if (AlphaMinus < -physL[N+1] + SoundSpeedL ) AlphaMinus = -physL[N+1] + SoundSpeedL;
        if (AlphaPlus  <  physR[N+1] + SoundSpeedR ) AlphaPlus  =  physR[N+1] + SoundSpeedR;
        if (AlphaMinus < -physR[N+1] + SoundSpeedR ) AlphaMinus = -physR[N+1] + SoundSpeedR;

        for (l=0; l<5; l++) {
          N = Sx*i+Sy*j+Sz*k+l;
          F_mid[N] = (AlphaPlus*FLX[N]+AlphaMinus*FRX[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
        }

        if (*max < AlphaPlus ) *max = AlphaPlus;
        if (*max < AlphaMinus) *max = AlphaMinus;
      }
    }
  }

  free (phys);
  free (phys_temp);
  free (physL);
  free (physR);
  free (FLX);
  free (FRX);
  free (UL);
  free (UR);
}

void riemansolverY(double *F_mid, double *U, double *max) {
  double *phys  = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *phys_temp= (double*) malloc (5*(X)*(Y+4)*(Z)*sizeof (double));
  double *physL = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *physR = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *FLY   = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *FRY   = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *UL    = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *UR    = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double SoundSpeedL, SoundSpeedR;											//L, SoundSpeedR, SoundSpeedLL, SoundSpeedRR;
  double AlphaPlus=0, AlphaMinus=0;
  int i, j, k, l, N, N1, N2;
  int Sx  = 5*Y*Z,     Sy  = 5*Z, Sz  = 5;
  int Sx1 = 5*(Y+1)*Z, Sy1 = 5*Z, Sz1 = 5;
  int Sx2 = 5*(Y+4)*Z, Sy2 = 5*Z, Sz2 = 5;

  *max=0;

  Ucalcinv (phys,U,(X)*(Y)*(Z));

/******************periodic********************/
  for (i=0;i<X;i++) {
    for (j=0;j<Y+4;j++) {
      for (k=0;k<Z;k++) {
        for (l=0;l<5;l++) {
          N = Sx2*i+Sy2*j+Sz2*k+l;
          if      (j<2) {
            phys_temp[N] = phys[Sx*i+Sy*(j+Y-2)+Sz*k+l];      // phys[N+Sy*(Y-2)];
          }
          else if (j>Y+1) {
            phys_temp[N] = phys[Sx*i+Sy*(j-Y-2)+Sz*k+l];      // phys[N-Sy*(Y+2)];
          }
          else phys_temp[N] = phys[Sx*i+Sy*(j-2)+Sz*k+l];     // phys[N-Sy*2];
        }
      }
    }
  }
/**********************************************/

/*******************outflow********************
  for (i=0;i<X;i++) {
    for (j=0;j<Y+4;j++) {
      for (k=0;k<Z;k++) {
        for (l=0;l<5;l++) {
          N = Sx2*i+Sy2*j+Sz2*k+l;
          if      (j<2) {
            phys_temp[N] = phys[Sx*i+Sy*(0)+Sz*k+l];      // phys[N+Sy*(Y-2)];
          }
          else if (j>Y+1) {
            phys_temp[N] = phys[Sx*i+Sy*(Y-1)+Sz*k+l];      // phys[N-Sy*(Y+2)];
          }
          else phys_temp[N] = phys[Sx*i+Sy*(j-2)+Sz*k+l];     // phys[N-Sy*2];
        }
      }
    }
  }
/***********************************************/

  for (i=0;i<X;i++) {
    for (j=0;j<Y+1;j++) {
      for (k=0;k<Z;k++) {
        for (l=0; l<5; l++) {
          N1 = Sx1*i+Sy1*j+Sz1*k+l;
          N2 = Sx2*i+Sy2*j+Sz2*k+l;
          physL[N1] = phys_temp[N2+Sy2]   + 0.5 * minmod (THETA*(phys_temp[N2+Sy2]   - phys_temp[N2]),     0.5*(phys_temp[N2+2*Sy2] - phys_temp[N2]),     THETA*(phys_temp[N2+2*Sy2] - phys_temp[N2+Sy2]));
          physR[N1] = phys_temp[N2+2*Sy2] - 0.5 * minmod (THETA*(phys_temp[N2+2*Sy2] - phys_temp[N2+Sy2]), 0.5*(phys_temp[N2+3*Sy2] - phys_temp[N2+Sy2]), THETA*(phys_temp[N2+3*Sy2] - phys_temp[N2+2*Sy2]));
        }
      }
    }
  }

  FluxCalcPY (FLY, physL,(X)*(Y+1)*(Z));
  FluxCalcPY (FRY, physR,(X)*(Y+1)*(Z));
  Ucalc     (UL,           physL,(X)*(Y+1)*(Z));
  Ucalc     (UR,           physR,(X)*(Y+1)*(Z));

  for (i=0;i<X;i++) {
    for (j=0;j<Y+1;j++) {
      for (k=0;k<Z;k++) {
        N = Sx1*i+Sy1*j+Sz1*k;
        AlphaPlus=0.;
        AlphaMinus=0.;
        SoundSpeedL = sqrt (GAMMA * physL[N+4] / physL[N+0]);
        SoundSpeedR = sqrt (GAMMA * physR[N+4] / physR[N+0]);
        if (AlphaPlus  <  physL[N+2] + SoundSpeedL ) AlphaPlus  =  physL[N+2] + SoundSpeedL;
        if (AlphaMinus < -physL[N+2] + SoundSpeedL ) AlphaMinus = -physL[N+2] + SoundSpeedL;
        if (AlphaPlus  <  physR[N+2] + SoundSpeedR ) AlphaPlus  =  physR[N+2] + SoundSpeedR;
        if (AlphaMinus < -physR[N+2] + SoundSpeedR ) AlphaMinus = -physR[N+2] + SoundSpeedR;

        for (l=0; l<5; l++) {
          N = Sx1*i+Sy1*j+Sz1*k+l;
          F_mid[N] = (AlphaPlus*FLY[N]+AlphaMinus*FRY[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
        }

        if (*max < AlphaPlus ) *max = AlphaPlus;
        if (*max < AlphaMinus) *max = AlphaMinus;
       }
    }
  }

  free (phys);
  free (phys_temp);
  free (physL);
  free (physR);
  free (FLY);
  free (FRY);
  free (UL);
  free (UR);
}

void riemansolverZ(double *F_mid, double *U, double *max) {
  double *phys  = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *phys_temp= (double*) malloc (5*(X)*(Y)*(Z+4)*sizeof (double));
  double *physL = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *physR = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *FLZ   = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *FRZ   = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *UL    = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *UR    = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double SoundSpeedL, SoundSpeedR;											//L, SoundSpeedR, SoundSpeedLL, SoundSpeedRR;
  double AlphaPlus=0, AlphaMinus=0;
  int i, j, k, l, N, N1, N2;
  int Sx = 5*Y*Z, Sy = 5*Z, Sz = 5;
  int Sx1 = 5*Y*(Z+1), Sy1 = 5*(Z+1), Sz1 = 5;
  int Sx2 = 5*Y*(Z+4), Sy2 = 5*(Z+4), Sz2 = 5;

  Ucalcinv (phys,U,(X)*(Y)*(Z));

/************************reflective*************************/
  *max=0;
  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z+4;k++) {
        for (l=0;l<5;l++) {
          N = Sx2*i+Sy2*j+Sz2*k+l;
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
  }
/**********************************************************/

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z+1;k++) {
        for (l=0; l<5; l++) {
          N1 = Sx1*i+Sy1*j+Sz1*k+l;
          N2 = Sx2*i+Sy2*j+Sz2*k+l;
          physL[N1] = phys_temp[N2+Sz2]   + 0.5 * minmod (THETA*(phys_temp[N2+Sz2]   - phys_temp[N2]),     0.5*(phys_temp[N2+2*Sz2] - phys_temp[N2]),     THETA*(phys_temp[N2+2*Sz2] - phys_temp[N2+Sz2]));
          physR[N1] = phys_temp[N2+2*Sz2] - 0.5 * minmod (THETA*(phys_temp[N2+2*Sz2] - phys_temp[N2+Sz2]), 0.5*(phys_temp[N2+3*Sz2] - phys_temp[N2+Sz2]), THETA*(phys_temp[N2+3*Sz2] - phys_temp[N2+2*Sz2]));
        }
      }
    }
  }

  FluxCalcPZ (FLZ,physL,(X)*(Y)*(Z+1));
  FluxCalcPZ (FRZ,physR,(X)*(Y)*(Z+1));
  Ucalc     (UL,           physL,(X)*(Y)*(Z+1));
  Ucalc     (UR,           physR,(X)*(Y)*(Z+1));

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z+1;k++) {
        N = Sx1*i+Sy1*j+Sz1*k;
        AlphaPlus=0;
        AlphaMinus=0;
        SoundSpeedL = sqrt (GAMMA * physL[N+4] / physL[N+0]);
        SoundSpeedR = sqrt (GAMMA * physR[N+4] / physR[N+0]);
        if (AlphaPlus  <  physL[N+3] + SoundSpeedL ) AlphaPlus  =  physL[N+3] + SoundSpeedL;
        if (AlphaMinus < -physL[N+3] + SoundSpeedL ) AlphaMinus = -physL[N+3] + SoundSpeedL;
        if (AlphaPlus  <  physR[N+3] + SoundSpeedR ) AlphaPlus  =  physR[N+3] + SoundSpeedR;
        if (AlphaMinus < -physR[N+3] + SoundSpeedR ) AlphaMinus = -physR[N+3] + SoundSpeedR;

        for (l=0; l<5; l++) {
          N = Sx1*i+Sy1*j+Sz1*k+l;
          F_mid[N] = (AlphaPlus*FLZ[N]+AlphaMinus*FRZ[N]-AlphaMinus*AlphaPlus*(UR[N]-UL[N]))/(AlphaPlus+AlphaMinus);
        }

        if (*max < AlphaPlus ) *max = AlphaPlus;
        if (*max < AlphaMinus) *max = AlphaMinus;

      }
    }
  }

  free (phys);
  free (phys_temp);
  free (physL);
  free (physR);
  free (FLZ);
  free (FRZ);
  free (UL);
  free (UR);
}

void Fluxsource (double *Source, double *U, double *dt) {

  double *phys      = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *Potential = (double*) malloc (  (X+2)*(Y+2)*(Z+2)*sizeof (double));
  double dx, dy, dz;
  double gx, gy, gz;
  Grid (&dx, &dy, &dz);
  double x, y, z;
  double GridRatioX = *dt/dx; 
  double GridRatioY = *dt/dy; 
  double GridRatioZ = *dt/dz; 
  int Sx  = (Y+2)*(Z+2),   Sy = (Z+2),   Sz = 1;
  int Sx1 = 5*(Y+1)*(Z+1),   Sy1 = 5*(Z+1),  Sz1 = 5;
  int i,j,k,l,Nf, N;

  Ucalcinv(phys, U, (X)*(Y)*(Z));

  for (i=0;i<X+2;i++) {
    for (j=0;j<Y+2;j++) {
      for (k=0;k<Z+2;k++) {
        x = XMIN + dx * (i-1);
        y = YMIN + dy * (j-1);
        z = ZMIN + dz * (k-1);
        N  = (i*Sx + j*Sy + k*Sz);
        Potential[N] = G * z;
      }
    }
  }


  Sx  = (Y)*(Z);   Sy = (Z);   Sz = 1;
  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        Sx  = (Y+2)*(Z+2);
        Sy = (Z+2);
        Sz = 1;
        N  = ((i+1)*Sx + (j+1)*Sy + ((k+1)*Sz));
        gx = 0;//(Potential[N + Sx] - Potential[N - Sx]) * GridRatioX / 2;
        gy = 0;//(Potential[N + Sy] - Potential[N - Sy]) * GridRatioY / 2;
        gz = G*(*dt);//(Potential[N + Sz] - Potential[N - Sz]) * GridRatioZ / 2 ;
        Sx = (Y)*(Z);
        Sy = (Z);
        Sz = 1;
        Nf = 5*(i*Sx + j*Sy + k*Sz);
        Source[Nf+0] = 0.;
        Source[Nf+1] = - gx * phys[Nf+0];
        Source[Nf+2] = - gy * phys[Nf+0];
        Source[Nf+3] = - gz * phys[Nf+0];
        Source[Nf+4] = - phys [Nf+0] * (gx * phys[Nf+1] + gy * phys[Nf+2] + gz * phys[Nf+3]);
      }
    }
  }

  free (Potential);
  free (phys);
}

// Advances the function in time:
void Advance(double *U_new,double *U_old, double *B, double *dt) {
  int i=0, j=0, k=0, l=0, N;								// counters

  double *phys   = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *FmidX  = (double*) malloc (5*(X+1)*(Y)*(Z)*sizeof (double));
  double *FmidY  = (double*) malloc (5*(X)*(Y+1)*(Z)*sizeof (double));
  double *FmidZ  = (double*) malloc (5*(X)*(Y)*(Z+1)*sizeof (double));
  double *LU     = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *U1     = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *U2     = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double *Source = (double*) malloc (5*(X)*(Y)*(Z)*sizeof (double));
  double max1, maxX=0, maxY=0, maxZ=0;
  int Sx  = 5*Y*Z,     Sy = 5*Z,      Sz = 5;
  int Sxx = 5*Y*Z,     Syx = 5*Z,     Szx = 5;
  int Sxy = 5*(Y+1)*Z, Syy = 5*Z,     Szy = 5;
  int Sxz = 5*Y*(Z+1), Syz = 5*(Z+1), Szz = 5;
  int N1, Nx, Ny, Nz;

  double  GridRatioX, GridRatioY, GridRatioZ;
  Grid(&GridRatioX, &GridRatioY, &GridRatioZ);
  GridRatioX = *dt/GridRatioX; 
  GridRatioY = *dt/GridRatioY; 
  GridRatioZ = *dt/GridRatioZ; 

  riemansolverX(FmidX,U_old,&max1);
  if (max1> maxX) maxX=max1;
  riemansolverY(FmidY,U_old,&max1);
  if (max1> maxY) maxY=max1;
  riemansolverZ(FmidZ,U_old,&max1);
  if (max1> maxZ) maxZ=max1;
  Fluxsource(Source, U_old, dt);

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0; l<5; l++) {
          N  = i*Sx + j*Sy + k*Sz + l;
          N1  = i*Sx + j*Sy + k*Sz;
          Nx = i*Sxx + j*Syx + k*Szx + l;
          Ny = i*Sxy + j*Syy + k*Szy + l;
          Nz = i*Sxz + j*Syz + k*Szz + l;
          LU[N] = - GridRatioX * (FmidX[Nx+Sxx] - FmidX[Nx]) - GridRatioY * (FmidY[Ny+Syy] - FmidY[Ny]) - GridRatioZ * (FmidZ[Nz+Szz] - FmidZ[Nz]);
          U1[N] = U_old[N] + LU[N];
         }
      }
    }
  }

  riemansolverX(FmidX,U1,&max1);
  if (max1> maxX) maxX=max1;
  riemansolverY(FmidY,U1,&max1);
  if (max1> maxY) maxY=max1;
  riemansolverZ(FmidZ,U1,&max1);
  if (max1> maxZ) maxZ=max1;
  Fluxsource(Source, U1, dt);

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0; l<5; l++) {
          N  = i*Sx + j*Sy + k*Sz + l;
          N1 = i*Sx + j*Sy + k*Sz;
          Nx = i*Sxx + j*Syx + k*Szx + l;
          Ny = i*Sxy + j*Syy + k*Szy + l;
          Nz = i*Sxz + j*Syz + k*Szz + l;
          LU[N] = - GridRatioX * (FmidX[Nx+Sxx] - FmidX[Nx]) - GridRatioY * (FmidY[Ny+Syy] - FmidY[Ny]) - GridRatioZ * (FmidZ[Nz+Szz] - FmidZ[Nz]);
          U2[N] = 0.75 * U_old[N] + 0.25 * U1[N] + 0.25 * LU[N];
        }
      }
    }
  }

  riemansolverX(FmidX,U2,&max1);
  if (max1> maxX) maxX=max1;
  riemansolverY(FmidY,U2,&max1);
  if (max1> maxY) maxY=max1;
  riemansolverZ(FmidZ,U2,&max1);
  if (max1> maxZ) maxZ=max1;
  Fluxsource(Source, U2, dt);

  for (i=0;i<X;i++) {
    for (j=0;j<Y;j++) {
      for (k=0;k<Z;k++) {
        for (l=0; l<5; l++) {
          N  = i*Sx + j*Sy + k*Sz + l;
          N1 = i*Sx + j*Sy + k*Sz;
          Nx = i*Sxx + j*Syx + k*Szx + l;
          Ny = i*Sxy + j*Syy + k*Szy + l;
          Nz = i*Sxz + j*Syz + k*Szz + l;
          LU[N] = - GridRatioX * (FmidX[Nx+Sxx] - FmidX[Nx]) - GridRatioY * (FmidY[Ny+Syy] - FmidY[Ny]) - GridRatioZ * (FmidZ[Nz+Szz] - FmidZ[Nz]);
          U_new[N] = (1./3.) * U_old[N] + (2./3.) * U2[N] + (2./3.) * LU[N] + Source[N];
        }
      }
    }
  }

  Ucalcinv(phys, U_new, X*Y*Z);

  AdvanceB(B, phys, *dt);

  Grid(&GridRatioX, &GridRatioY, &GridRatioZ);

  double dtX = 0.3 * GridRatioX / maxX;
  double dtY = 0.3 * GridRatioY / maxY;
  double dtZ = 0.3 * GridRatioZ / maxZ;

  if (dtY<dtX) *dt=dtY;
  else *dt=dtX;
  if (dtZ<*dt) *dt=dtZ;

  free (phys);
  free (FmidX);
  free (FmidY);
  free (FmidZ);
  free (LU);
  free (U1);
  free (U2);
  free (Source);
}

// Output:
void output_file1(double *U, double t, int stat) {
  int i=0, j, k, l, N;
  double xstep, ystep, zstep;
  Grid  (&xstep, &ystep, &zstep);
  int Sx = 5*Y*Z, Sy = 5*Z, Sz = 5;
  double xproj, yproj, zproj;

  double *phys = (double*) malloc (5*X*Y*Z*sizeof(double));
  Ucalcinv(phys, U, (X)*(Y)*(Z));

  if (stat==0) system("rm numerical.dat");
  FILE *full = fopen("numerical.dat", "a+");
  for (i=0; i<X; i++) {
    for (j=0; j<Y; j++) {
      for (k=0; k<Z; k++) {
        N = Sx*i+Sy*j+Sz*k;
//        fprintf(full, "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",XMIN + i*xstep,YMIN + j*ystep,ZMIN + k*zstep,t,phys[N+0],phys[N+1],phys[N+2],phys[N+3],phys[N+4]);
      }
    }
  }
  fprintf(full, "\n\n");
  fclose(full);
  if (stat==0) system("rm xyproj.dat");
  FILE *projxy = fopen("xyproj.dat", "a+");
  if (stat==0) system("rm yzproj.dat");
  FILE *projyz = fopen("yzproj.dat", "a+");
  if (stat==0) system("rm zxproj.dat");
  FILE *projzx = fopen("zxproj.dat", "a+");
  for (i=0; i<X; i++) {
    for (j=0; j<Y; j++) {
      for (k=0; k<Z; k++) {
        N = Sx*i+Sy*j+Sz*k;
        if (i==X/2 && j==Y/2) fprintf(projxy, "%f	%f %f %f %f %f\n",ZMIN + k*zstep,phys[N+0],phys[N+1],phys[N+2],phys[N+3],phys[N+4]);
        if (j==Y/2 && k==Z/2) fprintf(projyz, "%f	%f %f %f %f %f\n",XMIN + i*xstep,phys[N+0],phys[N+1],phys[N+2],phys[N+3],phys[N+4]);
        if (k==Z/2 && i==X/2) fprintf(projzx, "%f	%f %f %f %f %f\n",YMIN + j*ystep,phys[N+0],phys[N+1],phys[N+2],phys[N+3],phys[N+4]);
      }
    }
  }
  fprintf(projxy, "\n\n");
  fprintf(projyz, "\n\n");
  fprintf(projzx, "\n\n");

  fclose(projxy);
  fclose(projyz);
  fclose(projzx);

  free (phys);
}

  
void output_file2(double *U, double *B, double t){

  int i=0, j, k, N;
  double xstep, ystep, zstep;
  Grid  (&xstep, &ystep, &zstep);
  int Sx = 5*Y*Z, Sy = 5*Z, Sz = 5;
  
  char command[160];
  char name[160];
  double *phys = (double*) malloc (5*X*Y*Z*sizeof(double));
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
      fprintf(vel , "%f\t",B[i*3*Y*Z+j*3*Z+k*3+1]);
      //fprintf(vel  , "%f\t",phys[N+3]);
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


void output_file3(double *U, double *B, double t){
    
    int i=0, j, k, l, N;
    double xstep, ystep, zstep;
    Grid  (&xstep, &ystep, &zstep);
    int Sx = 5*Y*Z, Sy = 5*Z, Sz = 5;
    double xproj, yproj, zproj;
    char command[120];
    char name[120];
    
    double *phys = (double*) malloc (5*X*Y*Z*sizeof(double));
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
                N = Sx*i+Sy*j+Sz*k;
                fprintf(dens , "%d\t%d\t%d\t%f\n",i,j,k,phys[N]);
                fprintf(vel  , "%d\t%d\t%d\t%f\n",i,j,k,phys[N+3]);
                fprintf(press, "%d\t%d\t%d\t%f\n",i,j,k,phys[N+4]);
            }
        }
    }
    fclose(dens);
    fclose(vel);
    fclose(press);

    sprintf(command, "cp density_%d_%d_%d_3d.dat %dx%dx%d/density/%f.dat",X,Y,Z,X,Y,Z,t);
    system(command);
    //sprintf(command, "cp velocity_%d_%d_%d_3d.dat %dx%dx%d/velocity/%f.dat",X,Y,Z,X,Y,Z,t);
    //      system(command);
    //sprintf(command, "cp pressure_%d_%d_%d_3d.dat %dx%dx%d/pressure/%f.dat",X,Y,Z,X,Y,Z,t);
    //system(command);

    free (phys);
}

void Creat_folder(){
  char command[120];
  sprintf(command, "mkdir %dx%dx%d",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/velocity",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/density",X,Y,Z);
  system(command);
  sprintf(command, "mkdir %dx%dx%d/pressure",X,Y,Z);
  system(command);

  sprintf(command, "mkdir %dx%d",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/velocity",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/density",X,Z);
  system(command);
  sprintf(command, "mkdir %dx%d/pressure",X,Z);
  system(command);
}

int main (int argc, char **argv) {
  double *phys = (double*) malloc(5*X*Y*Z*sizeof(double));
  double *U    = (double*) malloc(5*X*Y*Z*sizeof(double));
  double *U_adv = (double*) malloc(5*X*Y*Z*sizeof(double));
  double *B = (double*) malloc(3*X*Y*Z*sizeof(double));
  double dt=0.00001;
  double t=0;
  int i, k=0, l=0;

  Creat_folder();//Creat folder to store data;

   
  // Initial conditions:
  init_water_oil(phys);
  init_B_field(B);
  Ucalc(U,phys, (X)*(Y)*(Z));
  printf ("tmax = %f\n",tmax);
//  system ("python mkplot.py");
#if 0
  timestamp_type time1, time2;
  get_timestamp(&time1);
  Advance(U_adv, U, &dt);
  get_timestamp(&time2);
  printf("time for Advance: %f s\n", timestamp_diff_in_seconds(time1,time2));
#endif
#if 1
  while (t<tmax) {
//    output_file1(U, t, j);
    t+=dt;
    Advance(U_adv, U, B, &dt);
    for ( i=0;i<5*X*Y*Z;i++) {
      U[i]=U_adv[i];
    }
//    j++;
    if (t>0.05 * k) {
      output_file3(U, B, t);
      output_file2(U, B, t);
      k++;
    }
    printf ("%d| t = %f\n",l,t);
    l++;
  }

  free (phys);
  free (U);
  free (U_adv);
  free (B);
#endif
  return 0;
}
