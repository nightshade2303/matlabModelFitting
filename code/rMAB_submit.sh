#!/bin/bash
#SBATCH -J rMABfit #job_name
#SBATCH -N 1                 # number of nodes
#SBATCH -c 8            # number of cores
#SBATCH --mem 16000          # memory for all cores
#SBATCH -t 0-12:00           # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o %x_%a_%A.%N.%j.out     # STDOUT
#SBATCH -e %x_%a_%A.%N.%j.err     # STDERR

# $1 is the fit date/folder name
 
###############################################################################
# Environment
###############################################################################

export MLM_LICENSE_FILE=27000@matlab-tah.serc.iisc.ernet.in

MATLAB_ROOT=/mnt/pve/Homes/MATLAB/R2025b
MATLAB_BIN=$MATLAB_ROOT/bin/matlab

# Respect Slurm CPU allocation
export MATLAB_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



###############################################################################
# Run MATLAB
###############################################################################

cd /mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code

$MATLAB_BIN -nodisplay -nosplash -nodesktop -r "fitRatDataRCwrapper('/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/data/','/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/$1/', ${SLURM_ARRAY_TASK_ID});exit"
