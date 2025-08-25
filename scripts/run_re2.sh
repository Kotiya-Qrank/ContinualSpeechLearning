#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task RE
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task transcribe
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR-RE  --task KS

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task RE
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task transcribe
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR-RE  --task KS

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task RE
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task transcribe
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR-RE  --task KS


CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs8/PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE.log  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task NONE   --dataset continual_task/KS-SID-ER-IC-SF-ASR-RE  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001  --temperature 1  --freeze_weights True
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task RE
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE  --task KS

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs8/WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE.log  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task NONE   --dataset continual_task/KS-SID-ER-IC-SF-ASR-RE  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS-SID-ER-IC-SF-ASR  --temperature 1  --freeze_weights True
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task RE
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task transcribe
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE  --task KS

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs8/WSL-00005-KS-SID-ER-IC-SF-ASR-RE.log  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task NONE   --dataset continual_task/KS-SID-ER-IC-SF-ASR-RE  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/WSL-00005-KS-SID-ER-IC-SF-ASR  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task RE
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task transcribe
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name WSL-00005-KS-SID-ER-IC-SF-ASR-RE  --task KS
