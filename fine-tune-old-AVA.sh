export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database_audio.yml

# python /scratch/map22-share/pyav/dihard.py \
#      --groundtruths test \
#      --justaudio \
#      --protocol AVA-AVD.SpeakerDiarization.data

python /vast/map22/map22-share/map22-share/headcam/code/headcam/dihard.py \
       --groundtruths test \
       --finetune \
       --epochs 2 \
       --protocol AVA-AVD.SpeakerDiarization.data


