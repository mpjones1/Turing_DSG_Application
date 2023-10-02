#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=251G #Min:82G Max:251G
#SBATCH --comment=CHCFHdG_and_scattering_parallel_3D_F32
#SBATCH --time=8:00:00 #18:00:00
#SBATCH --export=ALL
#SBATCH --mail-user=mpjones1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# blend parameters
phi0=0.5 # average composition
N=2700 # degree of polymerisation
chi=0.000765 # interaction parameter
ICn=0 # magnitude of initial flucutation from phi0
l=321 # number of lattice sites in each dimension
v0=1 # monomeric volume
lambda=4.472135955  # Kuhn length. Note: 4.472135955 - sqrt(20), 3.16227766017 - sqrt(10)

# simulation parameters
dt=0.00004 # temporal discretisation
dts=0.025 # time between snapshots of scattering
dx=0.20 # spatial discretisation
nsteps=1250000 # number of time steps
nsnaps=2001 # number of snapshots of scattering
nthreads1D=4 # number of threads to request
nblocks1D=32 # number of blocks to request. Note: 8,16,32 for dx=1,0.5,0.25 respectively

# run julia script
julia Noise_SD_Scattering_Parallel_3D_F32.jl $phi0 $N $chi $ICn $l $v0 $lambda $dt $dts $dx $nsteps $nsnaps $c $nthreads1D $nblocks1D
