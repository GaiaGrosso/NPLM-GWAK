#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH --gpus 1
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p iaifi_gpu   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o flk-NPLM-store-1D_saveD_myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e flk-NPLM-store-1D_saveD_myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01
# run code
#python nplm-flk-11D.py -s background -w 99 -t 1000 -d 60 -r 940 -l 0.000001 -m 1000
python nplm-flk-11D.py -s sghf -w 50 -t 1000 -d 60 -r 940 -l 0.000001 -m 1000
