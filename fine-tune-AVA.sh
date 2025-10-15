export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database_short.yml


python /scratch/map22-share/pyav/dihard.py \
       --groundtruths test \
       --finetune \
       --justvideo \
       --project videoonly \
       --name 2e_2relu_5bot_lr1em4 \
       --cdir /scratch/map22-share/pyav/testruns \
       --epochs 2 \
       --protocol AVA-AVD.SpeakerDiarization.data


