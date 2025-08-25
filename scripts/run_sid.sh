CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-0001-SID.log  --name FS-0001-SID  --task SID  --temperature 0.001  --dataset few_shot/VoxCeleb1_top10
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-0001-SID  --task SID  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-00001-SID.log  --name FS-00001-SID  --task SID  --temperature 0.0001  --dataset few_shot/VoxCeleb1_top10
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00001-SID  --task SID  --temperature 0.0001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/0001-SID.log  --name 0001-SID  --task SID  --temperature 0.001  --dataset VoxCeleb1_top10
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0001-SID  --task SID  --temperature 0.001

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/00001-SID.log  --name 00001-SID  --task SID  --temperature 0.0001  --dataset VoxCeleb1_top10
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00001-SID  --task SID  --temperature 0.0001

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/FS-1-SID.log  --name FS-1-SID  --task SID  --temperature 1  --dataset few_shot/VoxCeleb1_top10
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-SID  --task SID  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/1-SID.log  --name 1-SID  --task SID  --temperature 1  --dataset VoxCeleb1_top10
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 1-SID  --task SID  --temperature 1
