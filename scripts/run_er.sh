CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-0001-ER.log  --name FS-0001-ER  --task ER  --temperature 0.001  --dataset few_shot/IEMOCAP
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-0001-ER  --task ER  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-00001-ER.log  --name FS-00001-ER  --task ER  --temperature 0.0001  --dataset few_shot/IEMOCAP
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00001-ER  --task ER  --temperature 0.0001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/0001-ER.log  --name 0001-ER  --task ER  --temperature 0.001  --dataset IEMOCAP
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0001-ER  --task ER  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/00001-ER.log  --name 00001-ER  --task ER  --temperature 0.0001  --dataset IEMOCAP
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00001-ER  --task ER  --temperature 0.0001

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-1-ER.log  --name FS-1-ER  --task ER  --temperature 1  --dataset few_shot/IEMOCAP
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-ER  --task ER  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/1-ER.log  --name 1-ER  --task ER  --temperature 1  --dataset IEMOCAP
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 1-ER  --task ER  --temperature 1
