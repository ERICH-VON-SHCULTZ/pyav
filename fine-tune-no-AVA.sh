export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database.yml

python /scratch/map22-share/pyav/dihard.py \
      --groundtruths test \
      --justaudio \
      --protocol AVA-AVD.SpeakerDiarization.data



