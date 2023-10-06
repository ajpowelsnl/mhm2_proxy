module rm upcxx
module rm upcxx-gpu
module rm PrgEnv-intel
module rm PrgEnv-cray
module rm PrgEnv-gnu

module rm craype
module rm craype-mic-knl
module rm craype-haswell
module rm craype-x86-skylake

module load cgpu

module load PrgEnv-gnu
module load craype
module load craype-x86-skylake

module load cuda
module load cmake
module load git
module load upcxx-gpu

module list

which g++
which gcc
which nvcc

export OMP_NUM_THREADS=1
export MHM2_CMAKE_EXTRAS="-DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)"

# to build:
# salloc -C gpu -t 30 -q interactive -t 30 -c 8 -G 1
#  cmake $MHM2_CMAKE_EXTRAS -DCMAKE_INSTALL_PREFIX=${inst:=install} $src && make -j 8 all 

# example execute on 2 nodes, with 8 GPUs per node
# sbatch -C gpu -t 30  --exclusive -A m342 -G 16 -N 2 --wrap="./mhm2-builds/FixCoriGPUBuild/build-gpu-gnu-RelWithDebug/install/bin/mhm2.py -r arctic_sample_0.fq  -v"

# export GASNET_AM_CREDITS_PP=24

