#!/bin/bash
#SBATCH -J gearshifftK80
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=62000M # gpu2
#SBATCH --partition=gpu2
#SBATCH --exclusive
#SBATCH --array 1-2
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

CURDIR=~/cuda-workspace/gearshifft
RESULTSA=${CURDIR}/results/K80/cuda-8.0.44
RESULTSB=${CURDIR}/results/K80/clfft-2.12.2

FEXTENTS1D=$CURDIR/config/extents_1d_publication.conf
FEXTENTS2D=$CURDIR/config/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/config/extents_3d_publication.conf
#FEXTENTS=$CURDIR/config/extents_all_publication.conf
FEXTENTS=${CURDIR}/config/extents_capped_all_publication.conf

# default is cuda 8.0.44
if [ $k -eq 1 ]; then
    mkdir -p ${RESULTSA}
    srun --cpu-freq=medium --gpufreq=2505:823 $CURDIR/release/gearshifft_cufft -f $FEXTENTS -o $RESULTSA/cufft_gcc5.3.0_RHEL6.8.csv
fi
if [ $k -eq 2 ]; then
    mkdir -p ${RESULTSB}
    module purge
    module load boost/1.60.0-gnu5.3-intelmpi5.1
    module load clFFT/2.12.2-cuda8.0-gcc5.3
    srun --cpu-freq=medium --gpufreq=2505:823 $CURDIR/release/gearshifft_clfft -f $FEXTENTS -o $RESULTSB/clfft_gcc5.3.0_RHEL6.8.csv
fi
module list
nvidia-smi -q -d PERFORMANCE
