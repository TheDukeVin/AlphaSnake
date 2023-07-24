#!/bin/bash
#SBATCH -c 50 # Number of cores requested
#SBATCH -t 8000 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=60000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o snake_std.out # Standard out goes to this file
#SBATCH -e snake_std.err # Standard err goes to this filehostname
srun -l a.out