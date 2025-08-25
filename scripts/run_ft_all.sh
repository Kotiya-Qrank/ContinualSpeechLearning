CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS.log  --name FT-KS  --task KS   --dataset speech_commands  --base_model whisper-base
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS-SID.log  --name FT-KS-SID  --task NONE   --dataset VoxCeleb1_top10  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/FT-KS
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS-SID-ER.log  --name FT-KS-SID-ER  --task NONE   --dataset IEMOCAP  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/FT-KS-SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS-SID-ER-IC.log  --name FT-KS-SID-ER-IC  --task NONE   --dataset fluent  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/FT-KS-SID-ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS-SID-ER-IC-SF.log  --name FT-KS-SID-ER-IC-SF  --task NONE   --dataset SNIPS  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/FT-KS-SID-ER-IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs5/FT-KS-SID-ER-IC-SF-ASR.log  --name FT-KS-SID-ER-IC-SF-ASR  --task NONE   --dataset LibriSpeech  --base_model /datanfs2/wgt/TransformersWhisper/output5/whisper-base/FT-KS-SID-ER-IC-SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FT-KS-SID-ER-IC-SF-ASR  --task KS
