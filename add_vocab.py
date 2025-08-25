import argparse
import functools
import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers import WhisperForConditionalGeneration

from transformers_whisper.modeling_whisper import WhisperForConditionalGenerationWithWeightedSumLayer
from transformers_whisper import WhisperProcessor
from whisper_utils import CustomDataset, DataCollatorSpeechSeq2SeqWithPadding, print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("name", type=str, default="None", help="起个名字吧")
add_arg("base_model", type=str, default="whisper-base", help="Whisper的基础模型")
add_arg("processor_path", type=str, default="whisper-processor", help="编码器的路径，或者是huggingface上模型的名称")
add_arg("output_dir", type=str, default="output", help="训练保存模型的路径")
args = parser.parse_args()
print_arguments(args)

# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(os.path.join("pretrained_models", args.processor_path),
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# 修改模型词表
# tokens = ["<|SF|>"]
# processor.tokenizer.add_tokens(tokens, special_tokens=True)
# processor.save_pretrained("pretrained_models/whisper-processor.new")

tokens = ["<|speaker" + str(i) + "|>" for i in range(1, 1251)]

with open("data/SNIPS/slots.txt") as f:
    lines = f.readlines()
    labels = lines[1:]
    tokens.extend(["B-" + label.strip() for label in labels])
    tokens.extend(["E-" + label.strip() for label in labels])
processor.tokenizer.add_tokens(tokens, special_tokens=False)
processor.save_pretrained("pretrained_models/whisper-processor.final")

