# Speech Diarization Code Repository

This git repository contains the code used for speech diarization. For the environment file and checkpoints, please refer to `/scratch/map22-share/pyav` in the NYU HPC.

## Setup Instructions

### Prerequisites
1. Replace `YOUR_NETID` with your actual NetID
2. Replace `YOUR_API_KEY` with your WandB API key

### SBATCH Script

Use the following SBATCH file to run the code:

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=v13.80g
#SBATCH --output=./v13.out
#SBATCH --error=./v13.err

module purge

singularity exec --nv \
    --overlay /scratch/YOUR_NETID/pyav/pyav.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "

source /ext3/env.sh

cd /scratch/YOUR_NETID/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/YOUR_NETID/pyav/database.new2.yml
export WANDB_API_KEY=YOUR_API_KEY
export WANDB_INSECURE_DISABLE_SSL=true

python /scratch/YOUR_NETID/pyav/dihard.py \
       --groundtruths test \
       --finetune \
       --model 0 \
       --esize 384 \
       --project both13 \
       --name model_0_dihard_50e_lr1em4_2_helen \
       --cdir /scratch/YOUR_NETID/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data

"
```

## Important Notes

- **Environment**: The environment file and checkpoints are located in `/scratch/map22-share/pyav`
- **GPU Requirements**: This script requires 1 GPU with 80GB memory
- **Runtime**: Estimated runtime is 6 hours
- **Dependencies**: Uses CUDA 11.8.86 with cuDNN 8.7

## Configuration

Make sure to update the following before running:
- `YOUR_NETID`: Your actual NYU NetID
- `YOUR_API_KEY`: Your WandB API key for experiment tracking
