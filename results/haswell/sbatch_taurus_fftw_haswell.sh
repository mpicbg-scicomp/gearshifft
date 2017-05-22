#!/bin/bash
#SBATCH -J gearshifftFFTW
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=110:00:00
#SBATCH --mem=61991M
#SBATCH --partition=haswell64
#SBATCH --exclusive
#SBATCH --array 0-5,7-9
#SBATCH -o slurmcpu_array-%A_%a.out
#SBATCH -e slurmcpu_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

CURDIR=~/cuda-workspace/gearshifft
REL=$CURDIR/release
RESULTS=$CURDIR/results/haswell/fftw3.3.6pl1
mkdir -p ${RESULTS}

FEXTENTS1D=$CURDIR/config/extents_1d_publication.conf
FEXTENTS1DFFTW=$CURDIR/config/extents_1d_fftw.conf  # excluded a few very big ones
FEXTENTS2D=$CURDIR/config/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/config/extents_3d_publication.conf
FEXTENTS=$CURDIR/config/extents_all_publication.conf


if [ $k -eq 0 ]; then
#   srun $REL/gearshifft_fftw -f $FEXTENTS1D -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL6.8.1d.csv --rigor estimate # got killed by OS (OOM error)
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS1DFFTW -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL6.8.1d.csv --rigor estimate

elif [ $k -eq 1 ]; then
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS2D -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL6.8.2d.csv --rigor estimate

elif [ $k -eq 2 ]; then
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS3D -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL6.8.3d.csv --rigor estimate

elif [ $k -eq 3 ]; then
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS1DFFTW -o $RESULTS/fftw_wisdom_gcc5.3.0_RHEL6.8.1d.csv --rigor wisdom --wisdom_sp ./config/fftwf_wisdom_3.3.6-pl1.txt --wisdom_dp ./config/fftw_wisdom_3.3.6-pl1.txt

elif [ $k -eq 4 ]; then
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS2D -o $RESULTS/fftw_wisdom_gcc5.3.0_RHEL6.8.2d.csv --rigor wisdom --wisdom_sp ./config/fftwf_wisdom_3.3.6-pl1.txt --wisdom_dp ./config/fftw_wisdom_3.3.6-pl1.txt

elif [ $k -eq 5 ]; then
    srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS3D -o $RESULTS/fftw_wisdom_gcc5.3.0_RHEL6.8.3d.csv --rigor wisdom --wisdom_sp ./config/fftwf_wisdom_3.3.6-pl1.txt --wisdom_dp ./config/fftw_wisdom_3.3.6-pl1.txt

#elif [ $k -eq 6 ]; then
#     srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS1D -o $RESULTS/fftw_gcc5.3.0_RHEL6.8.1d.csv # takes too long, >7d
elif [ $k -eq 7 ]; then
     srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS2D -o $RESULTS/fftw_gcc5.3.0_RHEL6.8.2d.csv
elif [ $k -eq 8 ]; then
     srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS3D -o $RESULTS/fftw_gcc5.3.0_RHEL6.8.3d.csv
elif [ $k -eq 9 ]; then
     srun --cpu-freq=medium $REL/gearshifft_fftw -f $FEXTENTS1DFFTW -o $RESULTS/fftw_gcc5.3.0_RHEL6.8.1d.csv
fi

module list
