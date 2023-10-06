module rm upcxx
module rm PrgEnv-cray
module rm PrgEnv-gnu
module load PrgEnv-intel

module rm craype-mic-knl
# if these are loaded then the adept-sw library ends up static which causes the link to fail
module rm craype
module rm craype-haswell

module load esslurm
module load cuda
module rm cmake
module load cmake/3.18.2
module load git
module load upcxx-gpu

module list

export OMP_NUM_THREADS=1
export MHM2_CMAKE_EXTRAS="-DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc"
