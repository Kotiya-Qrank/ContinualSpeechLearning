#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS.log  --name PW-WSL-1-KS  --task KS  --temperature 1  --dataset speech_commands  --freeze_weights True  --base_model whisper-base
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID.log  --name PW-WSL-1-KS-SID  --task NONE  --temperature 1  --dataset continual_task/KS-SID  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER.log  --name PW-WSL-1-KS-SID-ER  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER-IC.log  --name PW-WSL-1-KS-SID-ER-IC  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC  --task KS  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER-IC-SF.log  --name PW-WSL-1-KS-SID-ER-IC-SF  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF  --task SF  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF  --task KS  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/PW-WSL-1-KS-SID-ER-IC-SF-ASR.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task transcribe  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task SF  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task KS  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF-ASR.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task transcribe  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task IC  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task ER  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR  --task KS  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#
#CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task transcribe  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task IC  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task ER  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005  --task KS  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#
#CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task transcribe  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task IC  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task ER  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001  --task KS  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#
#CUDA_VISIBLE_DEVICES=3 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task transcribe  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task IC  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task ER  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --task KS  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/PW-WSL-1-KS-SID-ER-IC-SF-ASR_100.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task transcribe  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task IC  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task ER  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_100  --task KS  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output5
