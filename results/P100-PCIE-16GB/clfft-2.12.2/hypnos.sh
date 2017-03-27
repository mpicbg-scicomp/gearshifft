#! /bin/bash
#PBS -N gearshifft_gpu0_p100_clfft
#PBS -l nodes=1:ppn=12
#PBS -l walltime=8:00:00
#PBS -q gpu-hotel
#PBS -o /home/steinb95/development/gearshifft/results/P100-PCIE-16GB/clfft-2.12.2/clfft_gcc5.3.0_ubuntu14.04.5.out
#PBS -e /home/steinb95/development/gearshifft/results/P100-PCIE-16GB/clfft-2.12.2/clfft_gcc5.3.0_ubuntu14.04.5.err

#how to launch
#$ qsub ./$0

source /opt/modules-3.2.6/Modules/3.2.6/init/bash
module load cuda/8.0 gcc/5.3.0 boost/1.63.0 amd-app-sdk//3.0.130.136 googletest/1.8.0 clfft/2.12.2 
module load numactl/2.0.7

#export CUDA_VISIBLE_DEVICES=0 #P100

numactl -N0 /home/steinb95/development/gearshifft/build/gearshifft_clfft -d 0:0 -f /home/steinb95/development/gearshifft/config/extents_all_publication.conf -o /home/steinb95/development/gearshifft/results/P100-PCIE-16GB/clfft-2.12.2/clfft_gcc5.3.0_ubuntu14.04.5.csv
