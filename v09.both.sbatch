#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=v09.both
#SBATCH --output=./v09.both.out
#SBATCH --error=./v09.both.err


module purge

singularity exec --nv --overlay /scratch/map22-share/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh

cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database_new_dino_small.yml

python /scratch/map22-share/pyav/dihard.py \
       --groundtruths test \
       --finetune \
       --model 0 \
       --esize 384 \
       --project both \
       --name model_0_dihard_50e_lr1em4_dino_small_v09 \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data

"

