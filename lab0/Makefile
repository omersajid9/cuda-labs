OBJ=lab0
LIBWBPATH=/projects/bche/labs/libwb

all:
	nvcc -std=c++11 -arch=sm_86 -I $(LIBWBPATH) -rdc=true -c $(OBJ).cu -o $(OBJ).o 
	nvcc -std=c++11 -arch=sm_86 -o $(OBJ) $(OBJ).o $(LIBWBPATH)/lib/libwb.so

clean:
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ) *.out *.err

run:
	sbatch job.slurm
