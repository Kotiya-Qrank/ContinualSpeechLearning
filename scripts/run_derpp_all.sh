CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS.log  --name DERPP-KS  --task KS   --dataset speech_commands  --base_model whisper-base
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS-SID.log  --name DERPP-KS-SID  --task SID   --dataset VoxCeleb1_top10  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/DERPP-KS  --method derpp
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS-SID-ER.log  --name DERPP-KS-SID-ER  --task ER   --dataset IEMOCAP  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/DERPP-KS-SID  --method derpp
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS-SID-ER-IC.log  --name DERPP-KS-SID-ER-IC  --task IC   --dataset fluent  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/DERPP-KS-SID-ER  --method derpp
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS-SID-ER-IC-SF.log  --name DERPP-KS-SID-ER-IC-SF  --task SF   --dataset SNIPS  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/DERPP-KS-SID-ER-IC  --method derpp
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF  --task KS

CUDA_VISIBLE_DEVICES=0 python whisper_finetune_cl.py>>/datanfs2/wgt/TransformersWhisper/logs4/DERPP-KS-SID-ER-IC-SF-ASR.log  --name DERPP-KS-SID-ER-IC-SF-ASR  --task transcribe   --dataset LibriSpeech  --base_model /datanfs2/wgt/TransformersWhisper/output4/whisper-base/DERPP-KS-SID-ER-IC-SF  --method derpp
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name DERPP-KS-SID-ER-IC-SF-ASR  --task KS
