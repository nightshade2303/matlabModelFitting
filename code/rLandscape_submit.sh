#!/bin/bash
#SBATCH -J fitLandscape            # job name
#SBATCH -N 1
#SBATCH --exclude=server1
#SBATCH --cpus-per-task=16         # number of cpus = parfor workers
#SBATCH --mem 24000                # memory for all cores (client holds ~0.6 GB of choices/rewards for 10k sets, plus one MATLAB worker per cpu)
#SBATCH -t 0-03:00                 # time (D-HH:MM); 10k sets ~ 3 CPU-hours -> ~15 min on 16 cpus
#SBATCH --export=ALL
#SBATCH -o /dev/null               # logs are redirected below into the run's own Jobs folder
#SBATCH -e /dev/null

# Runs testFitnessLandscape.m with the settings at the top of that script.
# Edit the settings there, then submit (from any directory):
#   sbatch /mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/rLandscape_submit.sh
# Results, figures and logs all go to output/sim/<yyyyMMdd>_landscape_<env>/,
# logs in its Jobs/ subfolder.

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
# Output folder for this run
###############################################################################

CODE_DIR=/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code
SIM_DIR=/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/sim

# read env ('unstr' / 'str') from the settings in testFitnessLandscape.m
ENV=$(sed -n "s/^env *= *'\([A-Za-z]*\)'.*/\1/p" $CODE_DIR/testFitnessLandscape.m | head -1)
if [ -z "$ENV" ]; then
    # nowhere better to log this: the run folder name depends on env
    echo "Could not read env from testFitnessLandscape.m" > $SIM_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_FAILED.err
    exit 1
fi

# same folder name testFitnessLandscape.m builds; passed to MATLAB so the
# results and logs land in the same place even if the job crosses midnight
export LANDSCAPE_SAVEFOLDER=$(date +%Y%m%d)_landscape_${ENV}
JOB_DIR=$SIM_DIR/$LANDSCAPE_SAVEFOLDER/Jobs
mkdir -p $JOB_DIR

# send everything from here on to the run's Jobs folder
exec > $JOB_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${SLURMD_NODENAME}.out \
    2> $JOB_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${SLURMD_NODENAME}.err

###############################################################################
# Run MATLAB
###############################################################################

cd $CODE_DIR

$MATLAB_BIN -nodisplay -nosplash -nodesktop -batch "testFitnessLandscape"
