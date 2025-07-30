#!/bin/bash
## Run with sbatch --array=1-3 launch.sh

#SBATCH --partition=long                                # Ask for unkillable job
#SBATCH --cpus-per-task=8                               # Ask for 2 CPUs
#SBATCH --ntasks=1                                      # Ask for 2 tasks per GPU 
#SBATCH --gres=gpu:1                                    # Ask for GPUs
#SBATCH --mem=96G                                       # Ask for 10 GB of CPU RAM
#SBATCH --time=11:55:00                                # The job will run for 3 hours
#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch
#SBATCH --no-requeue                                 # Do not requeue the job if it fails

module load cudatoolkit/12.1 miniconda/3 cuda/12.1.1/cudnn/9.3
conda activate roble

## srun is needed to run the multiple tasks per GPU
python mini_grp2.py r_seed=$SLURM_ARRAY_TASK_ID testing=false $ARGSS