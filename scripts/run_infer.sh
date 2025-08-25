#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name sum-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --temperature 1




#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-ER-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-IC-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-SF-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-ASR-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-ER-SID  --test_path data/IEMOCAP/test.json  --task ER
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-IC-SID  --test_path data/fluent/test.json  --task IC
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-SF-SID  --test_path data/SNIPS/test.json  --task SF
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name replay-ASR-SID  --test_path data/LibrSpeech/test.json  --task transcribe


#
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name sum4-SF-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name sum4-SF-SID  --test_path data/SNIPS/test.json  --task SF  --temperature 1  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name sum-ER-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name sum-ER-SID  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1


#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name replay3-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name sum3-KS-SID-ER-IC-SF-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name wsl-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf+wsl-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf+wsl-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf+wsl-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf+wsl-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf+wsl-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name lwf4-KS-SID-ER-IC-SF-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output

#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=1 python whisper_infer.py  --name none-KS-SID-ER-IC-SF-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name derpp-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name derpp-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name derpp-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name derpp-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name derpp-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF2  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF2  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF2  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF2  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF2  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF3-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name MW-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name MW-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name MW-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name MW-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF4  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF4  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF4  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF4  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF4  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=3 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr2e-6  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr2e-6  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr2e-6  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr2e-6  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr2e-6  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-6  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-6  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-6  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-6  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-6  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr1e-4  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr1e-4  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr1e-4  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr1e-4  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr1e-4  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/LibriSpeech/test.json  --task transcribe  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/SNIPS/test.json  --task SF  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/fluent/test.json  --task IC  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/IEMOCAP/test.json  --task ER  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/VoxCeleb1_top10/test.json  --task SID  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name PW-1-KS-SID-ER-IC-SF_lr5e-5-ASR  --test_path data/speech_commands/test.json  --task KS  --output_dir /datanfs2/wgt/TransformersWhisper/output  --temperature 1

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-ASR_lr1e-5  --test_path data/LibriSpeech/test.json  --task transcribe  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-SF_lr1e-5  --test_path data/SNIPS/test.json  --task SF  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-IC_lr1e-5  --test_path data/fluent/test.json  --task IC  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-ER_lr1e-5  --test_path data/IEMOCAP/test.json  --task ER  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-SID_lr1e-5  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-00005-KS_lr1e-5  --test_path data/speech_commands/test.json  --task KS  --temperature 0.0005

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-ASR_lr1e-5  --test_path data/LibriSpeech/test.json  --task transcribe  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-SF_lr1e-5  --test_path data/SNIPS/test.json  --task SF  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-IC_lr1e-5  --test_path data/fluent/test.json  --task IC  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-ER_lr1e-5  --test_path data/IEMOCAP/test.json  --task ER  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-SID_lr1e-5  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 00005-KS_lr1e-5  --test_path data/speech_commands/test.json  --task KS  --temperature 0.0005

#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-ASR_lr1e-5  --test_path data/LibriSpeech/test.json  --task transcribe  --temperature 0.0005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-SF_lr1e-5  --test_path data/SNIPS/test.json  --task SF  --temperature 0.005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-IC_lr1e-5  --test_path data/fluent/test.json  --task IC  --temperature 0.005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-ER_lr1e-5  --test_path data/IEMOCAP/test.json  --task ER  --temperature 0.005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-SID_lr1e-5  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 0.005
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name 0005-KS_lr1e-5  --test_path data/speech_commands/test.json  --task KS  --temperature 0.005
##
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/fluent/test.json  --task IC  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC-SF  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/IEMOCAP/test.json  --task ER  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER-IC  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER  --test_path data/VoxCeleb1_top10/test.json  --task SID  --temperature 1
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID-ER  --test_path data/speech_commands/test.json  --task KS  --temperature 1
#
#CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name FS-1-KS-SID  --test_path data/speech_commands/test.json  --task KS  --temperature 1

CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/re-tacred/test.json  --task RE
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/LibriSpeech/test.json  --task transcribe
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/SNIPS/test.json  --task SF
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/fluent/test.json  --task IC
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/IEMOCAP/test.json  --task ER
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/VoxCeleb1_top10/test.json  --task SID
CUDA_VISIBLE_DEVICES=0 python whisper_infer.py  --name multi_task2  --test_path data/speech_commands/test.json  --task KS