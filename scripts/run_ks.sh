CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-0001-KS.log  --name FS-0001-KS  --task KS  --temperature 0.001  --dataset few_shot/speech_commands
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-0001-KS  --task KS  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-00001-KS.log  --name FS-00001-KS  --task KS  --temperature 0.0001  --dataset few_shot/speech_commands
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00001-KS  --task KS  --temperature 0.0001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/0001-KS.log  --name 0001-KS  --task KS  --temperature 0.001  --dataset speech_commands
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0001-KS  --task KS  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/00001-KS.log  --name 00001-KS  --task KS  --temperature 0.0001  --dataset speech_commands
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00001-KS  --task KS  --temperature 0.0001

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-1-KS.log  --name FS-1-KS  --task KS  --temperature 1  --dataset few_shot/speech_commands
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/1-KS.log  --name 1-KS  --task KS  --temperature 1  --dataset speech_commands
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 1-KS  --task KS  --temperature 1
