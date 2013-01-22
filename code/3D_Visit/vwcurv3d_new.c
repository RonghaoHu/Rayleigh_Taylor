#include "visit_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 96
#define NY 96
#define NZ 288
#define FILE_SIZE 500



int main(int argc, char *argv[]) {
  //    float pts[NX*NY*NZ*3];
    FILE* file = fopen(argv[1], "r");
    //FILE* outFile = fopen(argv[3], "w");
    char str[FILE_SIZE];

    int dims[] = {NX, NY, NZ};
    /* Zonal and nodal variable data. */
    //    float zonal[NZ][NY][NX];
    //float nodal[NZ][NY][NX];
    /* Info about the variables to pass to visit_writer. */
    int nvars = 1;
    int vardims[] = {1};
    int centering[] = {1};
    int i,j,k;
    const char *varnames[] = {"zonal"};
    //float *vars[] = {(float *)zonal};

    float *pts = (float *) malloc(NX*NY*NZ*3*sizeof(float));
    
    float *zonal = (float *)malloc(NX*NY*NZ*sizeof(float));
    float *vars[] = {(float *)zonal};

    int x = 0;
    int y = 0;
    int z = 0;
    int index = 0;

    for (int i = 0; i < NZ; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NX; k++) {
                pts[index++] = k;
                pts[index++] = j;
                pts[index++] = i;
            }
        }
    }



    while(fgets(str, FILE_SIZE, file) != NULL) {
        char line[FILE_SIZE];
        strcpy(line, str);
        if ((index / 3) % 10000 == 0) {
            printf("Processed %d lines\n", index / 3);
        }
        if (strlen(line) >= 2) {
            int count = 0;
            for (char* token = strtok(line, "\t"); token != NULL; token = strtok(NULL, "\t")) {
                if (count < 3) {
                    double point_val = atoi(token);
                    if (count == 0) {
                        x = point_val;
                    } else if (count == 1) {
                        y = point_val;
                    } else {
                        z = point_val;
                    }
                    count++;
                } else {
                    double val = atof(token);
                    int pos = z * NY * NX + y * NX + x;
                    zonal[pos] = val;
                }
            }
            //fprintf(outFile, "%d\t%d\t%d\t%f\n", x, y, z, zonal[z][y][x]);
        }
    }
    int pos = 30 * NY * NX + 21 * NX + 50;
    printf("%f\n", zonal[pos]);


    /* Pass the data to visit_writer to write a binary VTK file. */
    write_curvilinear_mesh(argv[2], 1, dims, pts, nvars,vardims, centering, varnames, vars);
    free(zonal);
    free(pts);
}
