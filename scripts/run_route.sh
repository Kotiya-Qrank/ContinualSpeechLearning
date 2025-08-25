CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-KS.log  --name WSL-FS-1-KS  --task KS  --temperature 1  --freeze_weights True  --dataset speech_commands  --base_model whisper-base
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-SID.log  --name WSL-FS-1-SID  --task SID  --temperature 1  --freeze_weights True  --dataset VoxCeleb1_top10  --base_model whisper-base
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID  --task SID  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-ER.log  --name WSL-FS-1-ER  --task ER  --temperature 1  --freeze_weights True  --dataset IEMOCAP  --base_model whisper-base
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER  --task ER  --temperature 1

# KS-SID-ER
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-KS-SID.log  --name WSL-FS-1-KS-SID  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-SID  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-KS
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-SID  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-SID  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-KS-SID-ER.log  --name WSL-FS-1-KS-SID-ER  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-SID-ER  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-KS-SID
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task KS  --temperature 1

# KS-ER-SID
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-KS-ER.log  --name WSL-FS-1-KS-ER  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-ER  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-KS
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-ER  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-ER  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-KS-ER-SID.log  --name WSL-FS-1-KS-ER-SID  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-ER-SID  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-KS-ER
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-ER-SID  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-ER-SID  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-KS-ER-SID  --task KS  --temperature 1

# SID-KS-ER
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-SID-KS.log  --name WSL-FS-1-SID-KS  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/SID-KS  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-SID
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-KS  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-KS  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-SID-KS-ER.log  --name WSL-FS-1-SID-KS-ER  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-SID-ER  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-SID-KS
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-KS-ER  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-KS-ER  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-KS-ER  --task KS  --temperature 1

# SID-ER-KS
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-SID-ER.log  --name WSL-FS-1-SID-ER  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/SID-ER  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-SID
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-ER  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-ER  --task ER  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-SID-ER-KS.log  --name WSL-FS-1-SID-ER-KS  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/ER-SID-KS  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-SID-ER
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-ER-KS  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-ER-KS  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-SID-ER-KS  --task KS  --temperature 1

# ER-KS-SID
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-ER-KS.log  --name WSL-FS-1-ER-KS  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/ER-KS  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-ER
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-KS  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-KS  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-ER-KS-SID.log  --name WSL-FS-1-ER-KS-SID  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/KS-ER-SID  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-ER-KS
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-KS-SID  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-KS-SID  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-KS-SID  --task KS  --temperature 1

# ER-SID-KS
CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-ER-SID.log  --name WSL-FS-1-ER-SID  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/ER-SID  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-ER
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-SID  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-SID  --task ER  --temperature 1

CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/WSL-FS-1-ER-SID-KS.log  --name WSL-FS-1-ER-SID-KS  --task NONE  --temperature 1  --freeze_weights True  --dataset continual_task/ER-SID-KS  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/WSL-FS-1-ER-SID
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-SID-KS  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-SID-KS  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name WSL-FS-1-ER-SID-KS  --task KS  --temperature 1
