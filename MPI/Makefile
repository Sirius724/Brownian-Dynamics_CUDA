# Compiler and flags
NVCC = nvcc
MPIRUN = mpirun
CFLAGS = -I/home/appl/openmpi-4.0.5-ic1912/include -I ~/include
LDFLAGS = -L/home/appl/openmpi-4.0.5-ic1912/lib -L ~/lib
LIBS = -lmpi -lcudart

# Source files and executable
SRC = auto_Cell_list_mpi.cu
EXEC = mpi_auto_Cell_$(Temp).out
LOG = mpi_auto_Cell_$(Temp).out
compile_folder = compile

TEMP_VALUES = 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

EXECS = $(addprefix $(compile_folder)/mpi_auto_Cell_, $(TEMP_VALUES)).out

# Targets
all: $(EXEC)

$(compile_folder):
	mkdir -p $(compile_folder)

compile_file : $(SRC) 
	$(NVCC) $(SRC) -o $(compile_folder)/$(EXEC) $(CFLAGS) $(LDFLAGS) $(LIBS)

mpirun : 
	nohup $(MPIRUN) -n $(node) ./$(compile_folder)/$(EXEC) > $(compile_folder)/$(LOG) & 

$(compile_folder)/mpi_auto_Cell_%.out: $(SRC) | $(compile_folder)
	$(NVCC) $(SRC) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS)

mpirun_temp:
	for file in compile/mpi_auto_Cell_*.out; do
		nohup mpirun -n 2 ./${file} > ${file%.out}_log & 
		sleep 1
	done
