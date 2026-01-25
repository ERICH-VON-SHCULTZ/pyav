

# Multimodal Speaker Diarization Experiments

This repository contains the implementation for Audio-Visual Speaker Diarization using Pyannote. The system integrates visual information (video embeddings) into the standard audio pipeline at different stages using various fusion strategies.

## Model Architectures

We have implemented three distinct multimodal fusion strategies (Models 6, 7, and 8) to investigate the optimal stage for integrating visual cues.

### **Model 6: Late Fusion (Co-Attention)**

* **Architecture:** The audio stream is processed through the standard SincNet and LSTM layers first. The high-level audio features (post-LSTM) are then fused with video embeddings using a **Co-Attention Encoder**.
* **Logic:** This allows the model to learn temporal audio patterns first and uses visual information only to refine the final speaker embeddings before classification.
* **Key Feature:** Supports **LSTM Freezing** (see "Special Features").

### **Model 7: Early Fusion (Co-Attention)**

* **Architecture:** Visual integration happens at the raw feature level. The output of the SincNet (audio frames) and the video embeddings are fused using a **Co-Attention Encoder** *before* entering the LSTM.
* **Logic:** The LSTM processes a joint audio-visual representation, allowing it to model temporal dependencies on already fused multimodal data.

### **Model 8: Hybrid Fusion (Early Linear + Late Co-Attention)**

* **Architecture:** A dual-stage fusion approach.
1. **Early Stage:** Audio and Video features are projected using lightweight **Linear Layers** (and Dropout) and concatenated before the LSTM.
2. **Late Stage:** The output of the LSTM is fused again with video embeddings using a robust **Co-Attention Encoder**.


* **Logic:** The early linear fusion provides a "hint" of visual context to the LSTM without the computational cost of attention, while the late attention mechanism refines the final decision boundaries.

---

## Running Experiments (SBATCH)

Below are the SLURM scripts to train/finetune each model.

**Prerequisites:**

* Ensure your `Singularity` image and `PYANNOTE_DATABASE_CONFIG` paths are correct.
* `--finetune`: This flag is crucial. It loads the pre-trained segmentation model and adapts it to the new multimodal architecture.

### 1. Running Model 6 (Late Fusion)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_model6
#SBATCH --output=./logs/model6.out
#SBATCH --error=./logs/model6.err

module purge

singularity exec --nv --overlay /scratch/map22-share/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database.new2.yml

python /scratch/map22-share/pyav/dihard_integrated.py \
       --groundtruths test \
       --finetune \
       --model 6 \
       --esize 384 \
       --project both13 \
       --name model_6_late_fusion \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data
"

```

### 2. Running Model 7 (Early Fusion)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_model7
#SBATCH --output=./logs/model7.out
#SBATCH --error=./logs/model7.err

module purge

singularity exec --nv --overlay /scratch/map22-share/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database.new2.yml

python /scratch/map22-share/pyav/dihard_integrated.py \
       --groundtruths test \
       --finetune \
       --model 7 \
       --esize 384 \
       --project both13 \
       --name model_7_early_fusion \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data
"

```

### 3. Running Model 8 (Hybrid Fusion)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_model8
#SBATCH --output=./logs/model8.out
#SBATCH --error=./logs/model8.err

module purge

singularity exec --nv --overlay /scratch/map22-share/pyav/pyav.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
cd /scratch/map22-share/pyav

export PYANNOTE_DATABASE_CONFIG=/scratch/map22-share/pyav/database.new2.yml

python /scratch/map22-share/pyav/dihard_integrated.py \
       --groundtruths test \
       --finetune \
       --model 8 \
       --esize 384 \
       --project both13 \
       --name model_8_hybrid_fusion \
       --cdir /scratch/map22-share/pyav/experiments \
       --epochs 50 \
       --protocol AVA-AVD.SpeakerDiarization.data
"

```

---

## Special Features & Arguments


### LSTM Parameter Freezing

**Important Note for Model 6 (Late Fusion):**

To freeze the LSTM parameters during fine-tuning, you must execute the experiment using **`dihard.py`** instead of `dihard_integrated.py`.

The `dihard.py` script contains a dedicated function, `_apply_lstm_freezing`, which explicitly sets `requires_grad = False` for the LSTM layers when `model == 6` is selected. The integrated script (`dihard_integrated.py`) does not apply this freezing by default.

**Example Command Change:**

```bash
# To use LSTM freezing with Model 6, change the python script path:
python /scratch/map22-share/pyav/dihard.py \
       --model 6 \
       ... (rest of arguments)

```
### Key Arguments Explained

* `--model [int]`: Selects the architecture ID.
* `--finetune`: Enables the loading of pre-trained weights from `pyannote/segmentation-3.0` and prepares the model for transfer learning.
* `--esize [int]`: The embedding size of the input video features (e.g., 384 or 1024). This must match your generated `.npy` files.
* `--protocol`: The specific dataset protocol defined in your `database.yml`.
* `--groundtruths`: Path to the reference RTTM or 'test' to use the protocol's test set.
