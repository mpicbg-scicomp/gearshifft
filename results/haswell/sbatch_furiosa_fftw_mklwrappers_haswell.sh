#!/bin/bash
#SBATCH -J gearshifftFFTWMKL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128000M
#SBATCH --partition=long
#SBATCH --exclusive
#SBATCH --array 0-5
#SBATCH -o gearshifftcpu_array-%A_%a.out
#SBATCH -e gearshifftcpu_array-%A_%a.err

k=$SLURM_ARRAY_TASK_ID

CURDIR=/home/steinbac/development/gearshifft/
REL=$CURDIR/build-mkl
RESULTS=$CURDIR/results/haswell/fftwwrappers-gnu-2018u3

if [[ ! -e ${RESULTS} ]];
then
mkdir -p ${RESULTS}
fi


module load mkl-fftw/2018u3 parallel_studio_xe/2018u3 gcc/6.2.0

FEXTENTS1D=$CURDIR/config/extents_1d_publication.conf
FEXTENTS1DFFTW=$CURDIR/config/extents_1d_fftw.conf  # excluded a few very big ones
FEXTENTS2D=$CURDIR/config/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/config/extents_3d_publication.conf
FEXTENTS=$CURDIR/config/extents_all_publication.conf


if [ $k -eq 0 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS1DFFTW -o $RESULTS/fftw_estimate_gcc6.2.0_CENTOS7.5.1d.csv --rigor estimate

elif [ $k -eq 1 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS2D -o $RESULTS/fftw_estimate_gcc6.2.0_CENTOS7.5.2d.csv --rigor estimate

elif [ $k -eq 2 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS3D -o $RESULTS/fftw_estimate_gcc6.2.0_CENTOS7.5.3d.csv --rigor estimate

elif [ $k -eq 3 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS2D -o $RESULTS/fftw_gcc6.2.0_CENTOS7.5.2d.csv
elif [ $k -eq 4 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS3D -o $RESULTS/fftw_gcc6.2.0_CENTOS7.5.3d.csv
elif [ $k -eq 5 ]; then
    hostname
    srun $REL/gearshifft_fftwwrappers -f $FEXTENTS1DFFTW -o $RESULTS/fftw_gcc6.2.0_CENTOS7.5.1d.csv
fi

module list
