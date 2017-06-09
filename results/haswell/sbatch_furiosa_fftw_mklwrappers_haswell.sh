#!/bin/bash
#SBATCH -J gearshifftFFTWMKL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=100991M
#SBATCH --partition=long
#SBATCH --exclusive
#SBATCH --array 0-5
#SBATCH -o gearshifftcpu_array-%A_%a.out
#SBATCH -e gearshifftcpu_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

CURDIR=/home/steinbac/development/gearshifft/
REL=$CURDIR/build-mkl
RESULTS=$CURDIR/results/haswell/mklwrappers-2017u4
mkdir -p ${RESULTS}

module load fftw/3.3.6-pl1 mkl-fftw/2017u4 parallel_studio_xe/2017u4 gcc/5.3.0

FEXTENTS1D=$CURDIR/config/extents_1d_publication.conf
FEXTENTS1DFFTW=$CURDIR/config/extents_1d_fftw.conf  # excluded a few very big ones
FEXTENTS2D=$CURDIR/config/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/config/extents_3d_publication.conf
FEXTENTS=$CURDIR/config/extents_all_publication.conf


if [ $k -eq 0 ]; then
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS1DFFTW -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL7.2.1d.csv --rigor estimate

elif [ $k -eq 1 ]; then
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS2D -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL7.2.2d.csv --rigor estimate

elif [ $k -eq 2 ]; then
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS3D -o $RESULTS/fftw_estimate_gcc5.3.0_RHEL7.2.3d.csv --rigor estimate

elif [ $k -eq 3 ]; then
     srun $REL/gearshifft_fftwwrappers -f $FEXTENTS2D -o $RESULTS/fftw_gcc5.3.0_RHEL7.2.2d.csv
elif [ $k -eq 4 ]; then
     srun $REL/gearshifft_fftwwrappers -f $FEXTENTS3D -o $RESULTS/fftw_gcc5.3.0_RHEL7.2.3d.csv
elif [ $k -eq 5 ]; then
     srun $REL/gearshifft_fftwwrappers -f $FEXTENTS1DFFTW -o $RESULTS/fftw_gcc5.3.0_RHEL7.2.1d.csv
fi

module list
