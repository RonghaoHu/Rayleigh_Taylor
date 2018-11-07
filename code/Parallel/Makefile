hydro3d: hydro3d.cu
	nvcc -g -G $^ -o $@
	mv $@ ../bin
hydro3d_global: hydro3d_global.cu
	nvcc   -o $@ $^
