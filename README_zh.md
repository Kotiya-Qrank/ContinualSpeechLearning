# Continual Speech Learning with Fused Speech Features

代码来自 Interspeech 2025 文章: [Continual Speech Learning with Fused Speech Features](https://www.isca-archive.org/interspeech_2025/wang25u_interspeech.pdf)

### 环境设置

1. 创建并激活conda环境：
```bash
conda create -n Transper python==3.9
conda activate Transper
```

2. 安装PyTorch（请根据您的CUDA版本选择合适的安装命令）：
```bash
pip install torch torchvision torchaudio
```

3. 安装其他依赖包（还没上传requirements）：
```bash
pip install -r requirements.txt
```

4. 安装修改后的transformers库：
```bash
cd transformers
pip install .
```

### 模型准备

1. 下载Whisper预训练模型，并将其存放在`pretrained_models`目录中。

### 数据配置

在运行前，请将JSON文件中的数据路径`/data1/wgt/TransformersWhisper/data/speech_commands`更改为您本地的speech_commands数据路径。

### 运行命令

#### 无回放训练模型
```bash
python whisper_finetune_cl.py --name 00001-KS --task KS --temperature 0.0001 --dataset speech_commands  --base_model whisper-base
```

#### 有回放训练模型

Replay方法（由于在数据集中直接进行回放，因此需要将task设置为NONE）：
```bash
python whisper_finetune_cl.py --name Replay-KS-SID-ER-IC-SF-ASR --task NONE --dataset continual_task/KS-SID-ER-IC-SF-ASR --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID-ER-IC-SF
```

GFL_S方法（权重初始化为零）：
```bash
python whisper_finetune_cl.py --name WSL-00005-KS-SID-ER-IC-SF-ASR_lr1e-5 --task NONE --temperature 0.0005 --dataset continual_task/KS-SID-ER-IC-SF-ASR --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-00005-KS-SID-ER-IC-SF
```

GFL_D方法（需在代码中修改默认初始化参数）：
```bash
python whisper_finetune_cl.py --name PW-WSL-1-KS-SID-ER-IC-SF-ASR-RE --task NONE --dataset continual_task/KS-SID-ER-IC-SF-ASR-RE --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF-ASR --temperature 1 --freeze_weights True
```

#### 推理过程
```bash
python whisper_infer.py --name multi_task --test_path data/speech_commands/test.json --task KS
```
可选参数：
- `--prompt`: 是否在推理时添加prompt
- `--suppress`: 是否在推理时约束解码

