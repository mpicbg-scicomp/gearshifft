#!/bin/bash
#SBATCH -J gearshifftClFFT_CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=61991M
#SBATCH --partition=haswell64
#SBATCH --exclusive

module purge
module load boost/1.60.0-gnu5.3-intelmpi5.1
module load clFFT/2.12.2-cuda8.0-gcc5.3
module list

CURDIR=~/cuda-workspace/gearshifft

RESULTS=${CURDIR}/results/haswell/clfft-2.12.2
mkdir -p ${RESULTS}

FEXTENTS=${CURDIR}/config/extents_all_publication.conf

srun --cpu-freq=medium ${CURDIR}/release/gearshifft_clfft -f $FEXTENTS -d cpu -o $RESULTS/clfft_gcc5.3.0_RHEL6.8.csv
