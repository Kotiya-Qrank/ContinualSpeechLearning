# Continual Speech Learning with Fused Speech Features

The code is for Interspeech 2025 paper: [Continual Speech Learning with Fused Speech Features](https://www.isca-archive.org/interspeech_2025/wang25u_interspeech.pdf)

### Environment Setup

1. Create and activate conda environment:
```bash
conda create -n Transper python==3.9
conda activate Transper
```

2. Install PyTorch (choose the appropriate command based on your CUDA version):
```bash
pip install torch torchvision torchaudio
```

3. Install other required packages (requirements haven't uploaded yet):
```bash
pip install -r requirements.txt
```

4. Install the modified transformers library:
```bash
cd transformers
pip install .
```

### Model Preparation

1. Download the Whisper pre-trained model and place it in the `pretrained_models` directory.

### Data Configuration

Before running, please change the data path `/data1/wgt/TransformersWhisper/data/speech_commands` in the JSON files to your local speech_commands data path.

### Running Commands

#### Training Without Replay

```bash
python whisper_finetune_cl.py --name 00001-KS --task KS --temperature 0.0001 --dataset speech_commands
```

#### Training With Replay

Replay method (set task to NONE as replay is handled directly in the dataset):
```bash
python whisper_finetune_cl.py --name Replay-KS-SID-ER-IC-SF-ASR --task NONE --dataset continual_task/KS-SID-ER-IC-SF-ASR --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID-ER-IC-SF
```

GFL_S method (initialize weights to zero):
```bash
python whisper_finetune_cl.py --name WSL-00005-KS-SID-ER-IC-SF-ASR_lr1e-5 --task NONE --temperature 0.0005 --dataset continual_task/KS-SID-ER-IC-SF-ASR --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-00005-KS-SID-ER-IC-SF
```

GFL_D method (need to modify default initialization parameters in code):
```bash
python whisper_finetune_cl.py --name PW-WSL-1-KS-SID-ER-IC-SF-ASR-RE --task NONE --dataset continual_task/KS-SID-ER-IC-SF-ASR-RE --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF-ASR --temperature 1 --freeze_weights True
```

#### Inference Process
```bash
python whisper_infer.py --name multi_task --test_path data/speech_commands/test.json --task KS
```
Optional parameters:
- `--prompt`: Whether to add prompt during inference
- `--suppress`: Whether to constrain decoding during inference

