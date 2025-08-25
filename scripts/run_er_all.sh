CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS.log  --name Replay-KS  --task KS   --dataset speech_commands  --base_model whisper-base
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS-SID.log  --name Replay-KS-SID  --task NONE   --dataset continual_task/KS-SID  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS-SID-ER.log  --name Replay-KS-SID-ER  --task NONE   --dataset continual_task/KS-SID-ER  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS-SID-ER-IC.log  --name Replay-KS-SID-ER-IC  --task NONE   --dataset continual_task/KS-SID-ER-IC  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID-ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS-SID-ER-IC-SF.log  --name Replay-KS-SID-ER-IC-SF  --task NONE   --dataset continual_task/KS-SID-ER-IC-SF  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID-ER-IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs3/Replay-KS-SID-ER-IC-SF-ASR.log  --name Replay-KS-SID-ER-IC-SF-ASR  --task NONE   --dataset continual_task/KS-SID-ER-IC-SF-ASR  --base_model /datanfs2/wgt/TransformersWhisper/output3/whisper-base/Replay-KS-SID-ER-IC-SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name Replay-KS-SID-ER-IC-SF-ASR  --task KS
