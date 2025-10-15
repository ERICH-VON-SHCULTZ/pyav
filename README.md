sbatch file used to run the code:

```
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

singularity exec --nv --overlay /scratch/YOUR_NETID/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

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
replace **YOUR_NETID** to your actual netid

replace **YOUR_API_KEY** to your wandb api key
