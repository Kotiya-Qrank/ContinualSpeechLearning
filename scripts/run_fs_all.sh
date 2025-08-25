CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS.log  --name WSL-FS-1-KS  --task KS  --temperature 1  --dataset speech_commands  --freeze_weights True  --base_model whisper-base
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS-SID.log  --name WSL-FS-1-KS-SID  --task NONE  --temperature 1  --dataset continual_task/KS-SID  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS-SID-ER.log  --name WSL-FS-1-KS-SID-ER  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS-SID
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS-SID-ER-IC.log  --name WSL-FS-1-KS-SID-ER-IC  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS-SID-ER
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC  --task IC  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS-SID-ER-IC-SF.log  --name WSL-FS-1-KS-SID-ER-IC-SF  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS-SID-ER-IC
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF  --task SF  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF  --task IC  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=1 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/WSL-FS-1-KS-SID-ER-IC-SF-ASR.log  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task NONE  --temperature 1  --dataset continual_task/KS-SID-ER-IC-SF-ASR  --freeze_weights True  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/WSL-FS-1-KS-SID-ER-IC-SF
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task transcribe  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task SF  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task IC  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task ER  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task SID  --temperature 1
CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name WSL-FS-1-KS-SID-ER-IC-SF-ASR  --task KS  --temperature 1
