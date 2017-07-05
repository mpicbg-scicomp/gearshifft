#!/bin/bash
#SBATCH -J gearshifftP100
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=62000M # gpu2
#SBATCH --partition=test
#SBATCH --exclusive
#SBATCH --array 1-3
#SBATCH -o slurmgpu2_array-%A_%a.out
#SBATCH -e slurmgpu2_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

CURDIR=~/cuda-workspace/gearshifft
RESULTSA=${CURDIR}/results/P100-PCIE-16GB/cuda-8.0.61
RESULTSB=${CURDIR}/results/P100-PCIE-16GB/clfft-2.12.2
RESULTSC=${CURDIR}/results/P100-PCIE-16GB/cuda-9.0.69-beta

# FEXTENTS1D=$CURDIR/config/extents_1d_publication.conf
# FEXTENTS2D=$CURDIR/config/extents_2d_publication.conf
# FEXTENTS3D=$CURDIR/config/extents_3d_publication.conf
# FEXTENTS=$CURDIR/config/extents_all_publication.conf
FEXTENTS=${CURDIR}/config/extents_capped_all_publication.conf

module purge
export CUDA_VISIBLE_DEVICES=0

if [ $k -eq 1 ]; then
    module load boost/1.60.0-gnu5.3-intelmpi5.1 gdb git cmake clFFT/2.12.2-cuda8.0-gcc5.3
    module switch cuda/8.0.61_t1
    $CURDIR/release_t1/gearshifft_cufft -l
    mkdir -p ${RESULTSA}
    srun --cpu-freq=medium --gpufreq=715:1189 $CURDIR/release_t1/gearshifft_cufft -f $FEXTENTS -o $RESULTSA/cufft_gcc5.3.0_CentOS7.3.csv
fi
if [ $k -eq 2 ]; then
    module load boost/1.60.0-gnu5.3-intelmpi5.1 gdb git cmake cuda/9.0.69_beta
    $CURDIR/release_t1_cuda9/gearshifft_cufft -l
    mkdir -p ${RESULTSC}
    srun --cpu-freq=medium --gpufreq=715:1189 $CURDIR/release_t1_cuda9/gearshifft_cufft -f $FEXTENTS -o $RESULTSC/cufft_gcc5.3.0_CentOS7.3.csv
fi
if [ $k -eq 3 ]; then
    module load boost/1.60.0-gnu5.3-intelmpi5.1 gdb git cmake clFFT/2.12.2-cuda8.0-gcc5.3
    module switch cuda/8.0.61_t1
    $CURDIR/release/gearshifft_clfft -l
    mkdir -p ${RESULTSB}
    srun --cpu-freq=medium --gpufreq=715:1189 $CURDIR/release_t1/gearshifft_clfft -d gpu -f $FEXTENTS -o $RESULTSB/clfft_gcc5.3.0_CentOS7.3.csv
fi

nvidia-smi
nvidia-smi -q -d PERFORMANCE
