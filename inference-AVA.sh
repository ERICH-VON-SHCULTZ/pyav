export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database_short_old.yml

python /vast/map22/map22-share/map22-share/headcam/code/headcam/dihard.py \
       --groundtruths test \
       --protocol AVA-AVD.SpeakerDiarization.data

