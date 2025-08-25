CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-0001-ASR.log  --name FS-0001-ASR  --task transcribe  --temperature 0.001  --dataset few_shot/LibriSpeech
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-0001-ASR  --task transcribe  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-00001-ASR.log  --name FS-00001-ASR  --task transcribe  --temperature 0.0001  --dataset few_shot/LibriSpeech
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00001-ASR  --task transcribe  --temperature 0.0001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/0001-ASR.log  --name 0001-ASR  --task transcribe  --temperature 0.001  --dataset LibriSpeech
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0001-ASR  --task transcribe  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/00001-ASR.log  --name 00001-ASR  --task transcribe  --temperature 0.0001  --dataset LibriSpeech
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00001-ASR  --task transcribe  --temperature 0.0001

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-1-ASR.log  --name FS-1-ASR  --task transcribe  --temperature 1  --dataset few_shot/LibriSpeech
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-ASR  --task transcribe  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/1-ASR.log  --name 1-ASR  --task transcribe  --temperature 1  --dataset LibriSpeech
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 1-ASR  --task transcribe  --temperature 1
