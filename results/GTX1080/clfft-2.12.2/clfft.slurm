#!/bin/bash
#SBATCH -J gearshifft-clfft
#SBATCH -N 1
#SBATCH --array=0-6
#SBATCH -o clfft-%A_%a.log
#SBATCH -o clfft-%A_%a.err

module load clfft/2.12.2 boost/1.63.0

#please change this directory to where you have build gearshifft
BASE_DIR=${HOME}/development/gearshifft/build
CONF_DIR=${BASE_DIR}/../config/

#our system has 2 GPUs per server/node
for i in `seq 0 1`;
do
    ${BASE_DIR}/gearshifft_clfft -d "0:${i}" -o ${BASE_DIR}/gearshifft-clfft_${HOSTNAME}_${i}.csv -f ${CONF_DIR}/extents_all_publication.conf
done
