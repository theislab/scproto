#!/bin/bash

# Default job name and sbatch file name
JOB_NAME=${2:-job}
SBATCH_FILE=${1:-run.sbatch}

mkdir -p slurm-job-out/$JOB_NAME

sbatch --job-name=$JOB_NAME --output=slurm-job-out/$JOB_NAME/output.txt --error=slurm-job-out/$JOB_NAME/error.txt ./runs/sbatch_files/$SBATCH_FILE
