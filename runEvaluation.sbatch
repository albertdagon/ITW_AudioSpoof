#!/bin/bash
#SBATCH -JHGX_H100_Example               # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t3:00:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=ayou31@gatech.edu # E-mail address for notifications 

cd $HOME/scratch/ITW_AudioSpoof                  # Change to working directory

module load anaconda3                    # Load module dependencies
conda activate audio_spoof

module load gcc
module load cuda

srun python eval_samo.py --config ./configs/SSL_Test.conf
