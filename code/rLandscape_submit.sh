#!/bin/bash
#SBATCH -J fitLandscape            # job name
#SBATCH -N 1
#SBATCH --exclude=server1
#SBATCH --cpus-per-task=16         # number of cpus = parfor workers
#SBATCH --mem 24000                # memory for all cores (client holds ~0.6 GB of choices/rewards for 10k sets, plus one MATLAB worker per cpu)
#SBATCH -t 0-03:00                 # time (D-HH:MM); 10k sets ~ 3 CPU-hours -> ~15 min on 16 cpus
#SBATCH --export=ALL
#SBATCH -o %x_%j.%N.out            # STDOUT
#SBATCH -e %x_%j.%N.err            # STDERR

# Runs testFitnessLandscape.m with the settings at the top of that script.
# Edit the settings there, then submit from the code directory:
#   sbatch rLandscape_submit.sh
# Results and figures go to output/sim/<yyyyMMdd>_landscape_<env>/

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

$MATLAB_BIN -nodisplay -nosplash -nodesktop -batch "testFitnessLandscape"
