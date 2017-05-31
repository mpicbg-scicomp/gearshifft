#!/bin/bash
#SBATCH -J gearshifftFFTWWisdom
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --mem=128000M
#SBATCH --array 0-1
#SBATCH --partition=gpu
#SBATCH -o gearshifftFFTWWisdom-%A_%a.log

VVERS=3.3.6-pl1
VERS=fftw-${VVERS}

module load gcc/5.3.0 fftw/3.3.6-pl1-brdw

k=$SLURM_ARRAY_TASK_ID

if [ $k -eq 0 ]; then
    fftwf-wisdom -v -c -n -T 40 -o ~/development/gearshifft/config/fftwf_wisdom_${VVERS}_bdw-f32.txt

elif [ $k -eq 1 ]; then

    fftw-wisdom -v -c -n -T 40 -o ~/development/gearshifft/config/fftw_wisdom_${VVERS}_bdw-f64.txt
fi
