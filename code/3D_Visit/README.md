# Introduction

These are the codes for :
1. Generating the VTK files from raw data file
2. Grouping the VTK files into each timestamp 
3. Generating the file for playing the VTK files

# How to use it :

1. Change the dimension X Y Z in vwcurv3d_new.c
2. Run make Makefile to get the binary executable
3. Change the input and output directories in batch_write.sh
4. Run batch_write.sh with ./batch_write.sh
5. Run batch_group.sh to create the visit files for playing VTK files
6. Load the play.visit in visIt.

