CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-0001-SF.log  --name FS-0001-SF  --task SF  --temperature 0.001  --dataset few_shot/SNIPS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-0001-SF  --task SF  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-00001-SF.log  --name FS-00001-SF  --task SF  --temperature 0.0001  --dataset few_shot/SNIPS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00001-SF  --task SF  --temperature 0.0001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/0001-SF.log  --name 0001-SF  --task SF  --temperature 0.001  --dataset SNIPS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0001-SF  --task SF  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/00001-SF.log  --name 00001-SF  --task SF  --temperature 0.0001  --dataset SNIPS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00001-SF  --task SF  --temperature 0.0001

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-1-SF.log  --name FS-1-SF  --task SF  --temperature 1  --dataset few_shot/SNIPS
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-SF  --task SF  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/1-SF.log  --name 1-SF  --task SF  --temperature 1  --dataset SNIPS
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 1-SF  --task SF  --temperature 1
