#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=v13.both
#SBATCH --output=./v13.both.out
#SBATCH --error=./v13.both.err


module purge

singularity exec --nv --overlay /scratch/map22-share/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
2
cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database.new2.yml

python /scratch/map22-share/pyav/dihard.py \
       --groundtruths test \
       --finetune \
       --model 0 \
       --esize 384 \
       --project both13 \
       --name model_0_dihard_50e_lr1em4_2 \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data

"

