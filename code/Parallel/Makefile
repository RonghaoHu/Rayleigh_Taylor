rti2d: hydro3d.cu
	nvcc -DRTI2D -DMHD -Xcompiler="-fPIC" $^ -o $@ 
	mv $@ ../bin
hydro3d: hydro3d.cu
	nvcc -Xcompiler="-fPIC" $^ -o $@
	mv $@ ../bin
hydro3d_global: hydro3d_global.cu
	nvcc   -o $@ $^
