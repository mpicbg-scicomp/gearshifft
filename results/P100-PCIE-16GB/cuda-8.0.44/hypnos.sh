#! /bin/bash
#PBS -N gearshifft_gpu0_p100
#PBS -l nodes=1:ppn=12
#PBS -l walltime=12:00:00
#PBS -q gpu-hotel
#PBS -o /home/steinb95/development/gearshifft/results/P100/cuda-8.0.44/cufft_gcc5.3.0_ubuntu14.0.4.3.out
#PBS -e /home/steinb95/development/gearshifft/results/P100/cuda-8.0.44/cufft_gcc5.3.0_ubuntu14.0.4.3.err

#how to launch
#$ qsub ./$0

source /opt/modules-3.2.6/Modules/3.2.6/init/bash
module load cuda/8.0 gcc/5.3.0 boost/1.62.0

export CUDA_VISIBLE_DEVICES=0 #P100


/home/steinb95/development/gearshifft/build/gearshifft_cufft -f /home/steinb95/development/gearshifft/config/extents_all_publication.conf -o /home/steinb95/development/gearshifft/results/P100/cuda-8.0.44/cufft_gcc5.3.0_ubuntu14.0.4.3.csv -d 0
