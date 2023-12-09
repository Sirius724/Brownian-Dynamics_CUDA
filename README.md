# Brownian-Dynamics with CUDA

This code is written by CUDA and C++.
Your environment is in Linux. to compile, type "./measure_compile.sh".

I use the Verlet list method. If you want another nearest list method, you change that in the list update.

In this code, I measured the center of mass displacement(COM displacement), mean square displacement, and self-intermediate scattering function.
In MSD $\left<\Delta r(t)^2\right>$ and self-intermediate scattering function $F_s(q,t)$, we must consider COM displacement. so, I did that.

this code is based on Prof.Takeshi Kawasaki at Nagoya Univ (https://github.com/TakeshiKawasaki/GPU-for-MD)

## Compile for this code

In linux, just put "./mearsure_compile.sh". If there is a problem or error message, check the nvcc driver and header file directories.

## With MPI

You must check the directory of the MPI header file. and change it in the  compiled file.
Before compiling, first, in Linux, you check MPI library directory -> 'mpic++ -showme' -> '-I (address) -L (address) -lmpi' copy and paste in the compile file.

then, put "./mearsure_compile.sh" and execute the 'mpiexe.sh' also you must change the node number in compile file.

