#!/bin/bash
#SBATCH -J gearshifft-cufft
#SBATCH -N 1
#SBATCH --array=0-6
#SBATCH -o cufft-%A_%a.log
#SBATCH -o cufft-%A_%a.err

module load cuda/8.0.44 boost/1.63.0

#please change this directory to where you have build gearshifft
BASE_DIR=${HOME}/development/gearshifft/build
CONF_DIR=${BASE_DIR}/../config/

#our system has 2 GPUs per server/node
for i in `seq 0 1`;
do
    ${BASE_DIR}/gearshifft_cufft -d "${i}" -o ${BASE_DIR}/gearshifft-cufft_${HOSTNAME}_${i}.csv -f ${CONF_DIR}/extents_all_publication.conf
done
