cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database_new.yml

python /scratch/map22-share/pyav/dihard.py \
       --groundtruths test \
       --finetune \
       --model 0 \
       --project both \
       --name model_0_dihard_50e_lr1em4 \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data



