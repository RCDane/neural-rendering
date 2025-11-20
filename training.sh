#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpul40s
### -- set the job Name -- 
#BSUB -J neural_rendering_training
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
# ### -- specify that we need 4GB of memory per core/slot -- 
# #BSUB -R "rusage[mem=4GB]"
### -- specify that we want 1 GPU --
#BSUB -gpu "num=1"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
module load cuda/12.9.1
nvidia-smi

source ../.nr-env/bin/activate
python3 training_combined.py --output_folder output/run4 --training_config training_configs/default.yml > output.log