#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS_lr1e-5.log  --name PW-WSL-1-KS_lr1e-5  --task KS  --temperature 1  --dataset speech_commands  --base_model whisper-base  --learning_rate 1e-5  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS_lr1e-5  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-_lr1e-5.log  --name PW-WSL-1-KS-SID_lr1e-5  --task NONE  --temperature 1  --dataset continual_task/KS-SID  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/PW-WSL-1-KS_lr1e-5  --learning_rate 1e-5  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID_lr1e-5  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID_lr1e-5  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER_lr1e-5.log  --name PW-WSL-1-KS-SID-ER_lr1e-5  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/PW-WSL-1-KS-SID_lr1e-5  --learning_rate 1e-5  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER_lr1e-5  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER_lr1e-5  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER_lr1e-5  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC_lr1e-5.log  --name PW-WSL-1-KS-SID-ER-IC_lr1e-5  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/PW-WSL-1-KS-SID-ER_lr1e-5  --learning_rate 1e-5  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC_lr1e-5  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC_lr1e-5  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC_lr1e-5  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC_lr1e-5  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5.log  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/PW-WSL-1-KS-SID-ER-IC_lr1e-5  --learning_rate 1e-5  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task SF  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF_lr1e-5  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER-IC-SF-ASR_41.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --freeze_weights True
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task transcribe  --temperature 1
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task SF  --temperature 1
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task IC  --temperature 1
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_41  --task KS  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER-IC-SF-ASR_43.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task transcribe  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task SF  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_43  --task KS  --temperature 1
