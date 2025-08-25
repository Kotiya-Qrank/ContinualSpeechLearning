#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS.log  --name LWF-KS  --task KS   --dataset speech_commands  --base_model whisper-base  --method lwf
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS  --task KS
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS-SID.log  --name LWF-KS-SID  --task SID   --dataset VoxCeleb1_top10  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/LWF-KS  --method lwf
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID  --task KS
#
#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS-SID-ER.log  --name LWF-KS-SID-ER  --task ER   --dataset IEMOCAP  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/LWF-KS-SID  --method lwf
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER  --task KS

#CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS-SID-ER-IC.log  --name LWF-KS-SID-ER-IC  --task IC   --dataset fluent  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/LWF-KS-SID-ER  --method lwf
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS-SID-ER-IC-SF.log  --name LWF-KS-SID-ER-IC-SF  --task SF   --dataset SNIPS  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/LWF-KS-SID-ER-IC  --method lwf
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/LWF-KS-SID-ER-IC-SF-ASR.log  --name LWF-KS-SID-ER-IC-SF-ASR  --task transcribe   --dataset LibriSpeech  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/LWF-KS-SID-ER-IC-SF  --method lwf
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name LWF-KS-SID-ER-IC-SF-ASR  --task KS
