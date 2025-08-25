import argparse
import functools
import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers import WhisperForConditionalGeneration

from transformers_whisper.modeling_whisper import WhisperForConditionalGenerationWithWeightedSumLayer
from transformers_whisper import WhisperProcessor
from whisper_utils import CustomDataset, DataCollatorSpeechSeq2SeqWithPadding, print_arguments, add_arguments, compute_metrics

import torch
import random
import numpy as np

from models.lwf import *
from models.derpp import *

# 设置随机数种子
seed_value = 42

# 1. 设置 PyTorch 的随机数种子
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# 2. 设置 NumPy 的随机数种子
np.random.seed(seed_value)

# 3. 设置 Python 的随机数种子
random.seed(seed_value)

# 设置 cpu 数量
# cpu_cores0 = [0, 1, 2, 3, 4, 5, 6, 7]
cpu_cores1 = [8, 9, 10, 11, 12, 13, 14, 15]
cpu_cores2 = [16, 17, 18, 19, 20, 21, 22, 23]
cpu_cores3 = [24, 25, 26, 27, 28, 29, 30, 31]
cpu_cores4 = [32, 33, 34, 35, 36, 37, 38, 39]
cpu_cores5 = [40, 41, 42, 43, 44, 45, 46, 47]
cpu_cores6 = [48, 49, 50, 51, 52, 53, 54, 55]
cpu_cores7 = [56, 57, 58, 59, 60, 61, 62, 63]
cpu_cores8 = [64, 65, 66, 67, 68, 69, 70, 71]
cpu_cores = []
# cpu_cores.extend(cpu_cores0)
cpu_cores.extend(cpu_cores1)
cpu_cores.extend(cpu_cores2)
cpu_cores.extend(cpu_cores3)
cpu_cores.extend(cpu_cores4)
cpu_cores.extend(cpu_cores5)
# cpu_cores.extend(cpu_cores6)
# cpu_cores.extend(cpu_cores7)
# cpu_cores.extend(cpu_cores8)
os.sched_setaffinity(0, cpu_cores)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("name", type=str, default="None", help="起个名字吧")
add_arg("dataset", type=str, default=None, help="数据集名称")
add_arg("base_model", type=str, default="whisper-base", help="Whisper的基础模型")
add_arg("processor_path", type=str, default="whisper-processor.final",
        help="编码器的路径，或者是huggingface上模型的名称")
add_arg("weighted_sum_layer", type=bool, default=False, help="是否使用weighted sum layer")
add_arg("output_dir", type=str, default="/datanfs2/wgt/TransformersWhisper/output8", help="训练保存模型的路径")
# add_arg("warmup_steps", type=int, default=500, help="训练预热步数")
add_arg("logging_steps", type=int, default=50, help="打印日志步数")
add_arg("eval_steps", type=int, default=100, help="多少步数评估一次")
add_arg("save_steps", type=int, default=100, help="多少步数保存模型一次")
add_arg("num_workers", type=int, default=4, help="读取数据的线程数量")
add_arg("learning_rate", type=float, default=1e-4, help="学习率大小")
add_arg("min_audio_len", type=float, default=0.5, help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30, help="最大的音频长度，单位秒")
add_arg("fp16", type=bool, default=False, help="是否使用fp16训练模型")
add_arg("timestamps", type=bool, default=False, help="训练时是否使用时间戳数据")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("num_train_epochs", type=int, default=20, help="训练的轮数")
add_arg("language", type=str, default="English", help="设置语言，可全称也可简写，如果为None则训练的是多语言")
add_arg("freeze_encoder", type=bool, default=True, help="冻结编码器")
add_arg("freeze_weights", type=bool, default=False, help="冻结WSL权重")
add_arg("freeze_decoder", type=bool, default=False, help="冻结解码器")
add_arg("resume_from_checkpoint", type=str, default=None, help="恢复训练的检查点路径")
add_arg("batch_size", type=int, default=16, help="batch size")
add_arg("gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
add_arg("task", type=str, default="transcribe", help="模型的任务")
add_arg("temperature", type=float, default=-1, help="softmax的温度")
add_arg("method", type=str, default="none", help="持续学习方法", choices=["none", "er", "lwf", "ewc", "derpp"])
add_arg("predefined_weight", type=bool, default=False, help="是否预先设定weights")

args = parser.parse_args()
print_arguments(args)

# 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
processor = WhisperProcessor.from_pretrained(os.path.join("pretrained_models", args.processor_path),
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
tokenizer = processor.feature_extractor
dataset = args.dataset
if not args.dataset:
    if args.task == "KS":
        dataset = "speech_commands"
    elif args.task == "SID":
        dataset = "VoxCeleb1_top10"
    elif args.task == "ER":
        dataset = "IEMOCAP"
    elif args.task == "IC":
        dataset = "fluent"
    elif args.task == "SF":
        dataset = "SNIPS"
    elif args.task == "transcribe":
        dataset = "LibriSpeech"
    elif args.task == "RE":
        dataset = "re-tacred"

# 读取数据
train_dataset = CustomDataset(data_list_path=os.path.join("data", dataset, "train.json"),
                              processor=processor,
                              language=args.language)
valid_dataset = CustomDataset(data_list_path=os.path.join("data", dataset, "valid.json"),
                              processor=processor,
                              language=args.language)
if args.method == "derpp":
    task = args.task
    if args.task == "transcribe":
        task = "ASR"
    if args.task == "NONE":
        task = args.name.split("-")[-1]

    memory_dataset = CustomDataset("data/continual_task/" + ("KS-SID-ER-IC-SF-ASR-RE".split(task)[0] + task) + "/memory.json",
                                  processor=processor,
                                  language=args.language)
print(f"训练数据：{len(train_dataset)}，验证数据：{len(valid_dataset)}")

# 数据padding器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 获取模型
base_model = args.base_model
if args.base_model.split("-")[0] == "whisper":
    base_model = os.path.join("pretrained_models", args.base_model)
else:
    import json
    with open(args.base_model + "/trainer_state.json", "r") as f:
        trainer_state = json.load(f)
        base_model = trainer_state["best_model_checkpoint"]

if args.temperature > 0: args.weighted_sum_layer = True
if args.weighted_sum_layer:
    model = WhisperForConditionalGenerationWithWeightedSumLayer.from_pretrained(base_model)
    model.config.weighted_sum_layer = True
    model.model.encoder.weighted_sum_layer = True
    model.model.encoder.temperature = args.temperature
else:
    model = WhisperForConditionalGeneration.from_pretrained(base_model)

# 训练时设置为空
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 修改模型词表大小
model.resize_token_embeddings(len(processor.tokenizer))

if args.predefined_weight and base_model == "pretrained_models/whisper-base":
    if args.temperature == 0.0005:
        model.model.encoder.multi_weights = (
            torch.nn.Parameter(torch.tensor([
                [-3.6329e-04, -2.2242e-04, -1.5949e-04, -5.6149e-05, -2.9757e-04, 2.0240e-04,  6.2408e-04],
                [-2.2857e-04, -2.1294e-04, -3.0523e-04,  2.4889e-04,  1.2937e-04, 3.4960e-04, -4.3830e-05],
                [-2.7588e-04, -1.7018e-04, -1.0055e-04,  4.7533e-04,  3.9849e-05, 1.5688e-04,  7.9058e-05],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.]])))
    elif args.temperature == 0.0001:
        model.model.encoder.multi_weights = (
            torch.nn.Parameter(torch.tensor([
                [-1.2794e-04, -7.7738e-05, -4.0114e-05,  1.2115e-04, -2.8115e-04, 1.9903e-06,  2.0890e-04],
                [-1.0597e-05, -5.9521e-05, -2.1257e-04,  1.5180e-04, -7.8035e-05, 1.7350e-04, -9.6737e-05],
                [-2.0048e-04, -1.6915e-04, -1.2826e-04,  2.0661e-04, -1.4163e-04, 1.0613e-04,  9.6497e-05],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.]])))
    elif args.temperature == 1:
        model.model.encoder.multi_weights = (
            torch.nn.Parameter(torch.stack([
                torch.tensor([-0.0014, -0.0016, -0.0014, -0.0008, 0.0009, 0.0016, 0.0022]) / 0.01,
                torch.tensor([-2.2857e-04, -2.1294e-04, -3.0523e-04,  2.4889e-04,  1.2937e-04,  3.4960e-04, -4.3830e-05]) / 0.0005,
                torch.tensor([-3.7809e-05, -3.2926e-05, -2.5110e-05, 3.6659e-05, -4.7529e-05,  1.5444e-05, 2.8660e-05]) / 0.00001,
                torch.tensor([2.1853e-05, -3.2397e-05, -1.0472e-05, 3.7024e-05, -3.8597e-04,  -3.7812e-04, 1.4447e-04]) / 0.00005,
                torch.tensor([-1.2563e-06, 1.3448e-05, 1.0972e-05, -1.4510e-04, -1.8513e-04,  -1.3872e-04, 5.2880e-05]) / 0.00001,
                torch.tensor([-100., -100., -100., -100., -100., -100., 100.]) / 1,
            ], dim=0)))

# # 少样本
# model.model.encoder.multi_weights = (
#     torch.nn.Parameter(torch.stack([
#         torch.tensor([-5.1612e-04, -4.3862e-04, -4.1660e-04, -6.0471e-05, 2.7277e-04, 5.0754e-04, 6.3150e-04]) / 0.01,
#         torch.tensor([-1.6901e-04, -2.2255e-04, -2.8181e-04, -2.3166e-05, 6.6982e-05, 3.0708e-04, 1.8911e-04]) / 0.0005,
#         torch.tensor([-4.6760e-06, -3.8341e-05, -3.9812e-05, 2.9883e-05, -3.1938e-05, 6.7369e-06, 3.0017e-05]) / 0.00001,
#         torch.tensor([3.9441e-05, -9.3581e-05, -8.6563e-05, 7.9744e-05, -1.6259e-04, -6.6806e-05, 1.2090e-04]) / 0.00005,
#         torch.tensor([-1.5472e-06, 1.2440e-06, 1.0315e-05, -5.6472e-05, -7.4947e-05, -5.2912e-05, 4.1244e-05]) / 0.00001,
#         torch.tensor([-7.7366e-05, -1.8585e-04, -1.9319e-04, -2.0984e-04, -2.8158e-04, -3.3971e-05, 3.8898e-04]) / 0.0001,
#     ], dim=0)))
# 少样本2
# model.model.encoder.multi_weights = (
#     torch.nn.Parameter(torch.stack([
#         torch.tensor([-4.1988e-04, -2.1012e-04, -1.5714e-04, -2.6355e-05, -1.4056e-04, 2.6680e-04, 5.6261e-04]) / 0.0005,
#         torch.tensor([-5.7761e-04, -1.8662e-04, -2.5863e-04,  2.7087e-04,  4.7301e-04, 4.1421e-04, -2.6954e-05]) / 0.0005,
#         torch.tensor([-3.5279e-04, -1.8213e-04, -1.5859e-04,  5.0354e-04,  1.7250e-04, 1.3961e-04,  3.9025e-05]) / 0.0005,
#         torch.tensor([-0.0004, -0.0002, -0.0003, -0.0004, -0.0007, -0.0002,  0.0012]) / 0.0005,
#         torch.tensor([-0.0006, -0.0005, -0.0006, -0.0005, -0.0010, -0.0009,  0.0016]) / 0.0005,
#         torch.tensor([-5.1270e-04, -8.1919e-04, -8.3196e-04, -7.5622e-04, -6.1962e-04, -9.5008e-05,  1.4852e-03]) / 0.0005,
#     ], dim=0)))
# 少样本 1e-4 0.0005
if args.name == "WSL-FS-1-KS" or args.name == "WSL-FS-1-SID" or args.name == "WSL-FS-1-ER":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0013, -0.0013, -0.0013, -0.0011,  0.0008,  0.0006,  0.0020]) / 0.0005,
            torch.tensor([-0.0018, -0.0014, -0.0014, -0.0002,  0.0033,  0.0005, -0.0006]) / 0.0005,
            torch.tensor([-0.0011, -0.0011, -0.0012, -0.0005,  0.0015,  0.0006,  0.0015]) / 0.0005,
            torch.tensor([-0.0011, -0.0014, -0.0014, -0.0015, -0.0014, -0.0009,  0.0033]) / 0.0005,
            torch.tensor([-0.0009, -0.0011, -0.0012, -0.0011, -0.0013, -0.0011,  0.0027]) / 0.0005,
            torch.tensor([-6.0598e-04, -8.9142e-04, -8.6775e-04, -8.3087e-04, -5.6933e-04, 5.6311e-05,  1.5504e-03]) / 0.0005,
        ], dim=0)))

# # 全样本
# model.model.encoder.multi_weights = (
#     torch.nn.Parameter(torch.stack([
#         torch.tensor([-0.0014, -0.0016, -0.0014, -0.0008, 0.0009, 0.0016, 0.0022]) / 0.01,
#         torch.tensor([-2.2857e-04, -2.1294e-04, -3.0523e-04,  2.4889e-04,  1.2937e-04,  3.4960e-04, -4.3830e-05]) / 0.0005,
#         torch.tensor([-3.7809e-05, -3.2926e-05, -2.5110e-05, 3.6659e-05, -4.7529e-05,  1.5444e-05, 2.8660e-05]) / 0.00001,
#         torch.tensor([2.1853e-05, -3.2397e-05, -1.0472e-05, 3.7024e-05, -3.8597e-04,  -3.7812e-04, 1.4447e-04]) / 0.00005,
#         torch.tensor([-0.0007, -0.0009, -0.0011, -0.0015, -0.0026, -0.0012,  0.0023]) / 0.001,
#         torch.tensor([-100., -100., -100., -100., -100., -100., 100.]) / 1,
#     ], dim=0)))
# 全样本 1e-4 0.0005
if args.name == "PW-WSL-1-KS" or args.name == "PW-WSL-1-SID" or args.name == "PW-WSL-1-ER":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013,  0.0006, -0.0002,  0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014,  0.0016,  0.0024,  0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011,  0.0006,  0.0010,  0.0010,  0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009,  0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010,  0.0019]) / 0.0005,
            torch.tensor([-0.0008, -0.0012, -0.0012, -0.0012, -0.0010, -0.0004,  0.0021]) / 0.0005,
        ], dim=0)))
if args.name == "PW-WSL-1-KS-SID-ER-IC-SF-ASR_0005":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013,  0.0006, -0.0002,  0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014,  0.0016,  0.0024,  0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011,  0.0006,  0.0010,  0.0010,  0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009,  0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010,  0.0019]) / 0.0005,
            torch.tensor([-0.0055, -0.0069, -0.0070, -0.0078, -0.0053, -0.0008,  0.0101]) / 0.005,
        ], dim=0)))
if args.name == "PW-WSL-1-KS-SID-ER-IC-SF-ASR_0001":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013,  0.0006, -0.0002,  0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014,  0.0016,  0.0024,  0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011,  0.0006,  0.0010,  0.0010,  0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009,  0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010,  0.0019]) / 0.0005,
            torch.tensor([-0.0015, -0.0021, -0.0021, -0.0022, -0.0018, -0.0010,  0.0038]) / 0.001,
        ], dim=0)))
if args.name == "PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013,  0.0006, -0.0002,  0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014,  0.0016,  0.0024,  0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011,  0.0006,  0.0010,  0.0010,  0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009,  0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010,  0.0019]) / 0.0005,
            torch.tensor([-0.0003, -0.0003, -0.0004, -0.0004, -0.0004, -0.0002,  0.0007]) / 0.0001,
        ], dim=0)))
if args.name == "PW-WSL-1-KS-SID-ER-IC-SF-ASR_100":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013,  0.0006, -0.0002,  0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014,  0.0016,  0.0024,  0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011,  0.0006,  0.0010,  0.0010,  0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009,  0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010,  0.0019]) / 0.0005,
            torch.tensor([-100, -100, -100, -100, -100, -100, 100]) / 1,
        ], dim=0)))

# 全样本 1e-5 0.0005
if args.name == "PW-WSL-1-KS_lr1e-5":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-3.5991e-04, -2.1107e-04, -1.4693e-04, -5.3523e-05, -2.1837e-04, 2.4982e-04, 5.8946e-04]) / 0.0005,
            torch.tensor([-0.0011, -0.0003, -0.0004, 0.0006, 0.0008, 0.0005, -0.0003]) / 0.0005,
            torch.tensor([-0.0010, -0.0005, -0.0003, 0.0006, 0.0006, 0.0003, 0.0005]) / 0.0005,
            torch.tensor([-0.0012, -0.0008, -0.0007, -0.0011, -0.0009, -0.0011, 0.0020]) / 0.0005,
            torch.tensor([-0.0005, -0.0005, -0.0007, -0.0006, -0.0015, -0.0011, 0.0016]) / 0.0005,
            torch.tensor([-0.0007, -0.0011, -0.0012, -0.0012, -0.0012, -0.0007, 0.0021]) / 0.0005,
        ], dim=0)))

# 预定义RE实验
if args.name == "PW-WSL-1-KS-SID-ER-IC-SF-ASR_00001-RE" or args.name == "PW-WSL-1-KS-SID-ER-IC-RE" or args.name == "PW-WSL-1-KS-SID-ER-IC-SF-RE" or args.name == "PW-WSL-1-KS-SID-ER-IC-ASR_00001" or args.name == "PW-WSL-1-KS-SID-ER-IC-ASR" or args.name == "PW-WSL-1-KS-SID-ER-IC-ASR-RE" or args.name == "PW-WSL-1-KS-SID-ER-IC-ASR_00001-RE" or args.name == "DERPP-PW-WSL-1-KS-SID-ER-IC-SF-ASR-RE":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013, 0.0006, -0.0002, 0.0026]) / 0.0005,
            torch.tensor([-0.0020, -0.0016, -0.0014, 0.0016, 0.0024, 0.0001, -0.0009]) / 0.0005,
            torch.tensor([-0.0014, -0.0013, -0.0011, 0.0006, 0.0010, 0.0010, 0.0012]) / 0.0005,
            torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009, 0.0030]) / 0.0005,
            torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010, 0.0019]) / 0.0005,
            # torch.tensor([-0.0008, -0.0012, -0.0012, -0.0012, -0.0010, -0.0004,  0.0021]) / 0.0005,  # ASR
            torch.tensor([-0.0003, -0.0003, -0.0004, -0.0004, -0.0004, -0.0002, 0.0007]) / 0.0001,  # ASR
            torch.tensor([-0.0006, -0.0010, -0.0011, -0.0012, -0.0011, -0.0007,  0.0024]) / 0.0005,
        ], dim=0)))

# 少样本RE实验[-0.0004, -0.0007, -0.0008, -0.0007, -0.0005, -0.0003,  0.0019]
if args.name == "WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE" or args.name == "DERPP-WSL-FS-1-KS-SID-ER-IC-SF-ASR-RE":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0013, -0.0013, -0.0013, -0.0011, 0.0008, 0.0006, 0.0020]) / 0.0005,
            torch.tensor([-0.0018, -0.0014, -0.0014, -0.0002, 0.0033, 0.0005, -0.0006]) / 0.0005,
            torch.tensor([-0.0011, -0.0011, -0.0012, -0.0005, 0.0015, 0.0006, 0.0015]) / 0.0005,
            torch.tensor([-0.0011, -0.0014, -0.0014, -0.0015, -0.0014, -0.0009, 0.0033]) / 0.0005,
            torch.tensor([-0.0009, -0.0011, -0.0012, -0.0011, -0.0013, -0.0011, 0.0027]) / 0.0005,
            torch.tensor([-6.0598e-04, -8.9142e-04, -8.6775e-04, -8.3087e-04, -5.6933e-04, 5.6311e-05, 1.5504e-03]) / 0.0005,
            torch.tensor([-0.0004, -0.0007, -0.0008, -0.0007, -0.0005, -0.0003,  0.0019]) / 0.0005,
        ], dim=0)))

# 可训练RE实验
if args.name == "WSL-00005-KS-SID-ER-IC-SF-ASR-RE" or args.name == "DERPP-WSL-00005-KS-SID-ER-IC-SF-ASR-RE":
    model.model.encoder.multi_weights = (
        torch.nn.Parameter(torch.stack([
            torch.tensor([-0.0054, -0.0055, -0.0054, -0.0051, -0.0044, -0.0041, 0.0062]),
            torch.tensor([-0.0081, -0.0081, -0.0080, -0.0073, 0.0024, 0.0010, -0.0057]),
            torch.tensor([-0.0073, -0.0058, -0.0045, 0.0010, 0.0006, 0.0032, -0.0004]),
            torch.tensor([-0.0039, -0.0040, -0.0041, -0.0045, 0.0018, -0.0010, 0.0023]),
            torch.tensor([-0.0013, -0.0013, -0.0018, -0.0030, -0.0039, -0.0035, 0.0031]),
            torch.tensor([-0.0004, -0.0014, -0.0016, -0.0024, -0.0018, -0.0015, 0.0027]),
            torch.tensor([0., 0., 0., 0., 0., 0., 0.]),
        ], dim=0)))


    # # 平均
# model.model.encoder.multi_weights = (
#     torch.nn.Parameter(torch.stack([
#         torch.tensor([-0.0014, -0.0016, -0.0014, -0.0008, 0.0009, 0.0016, 0.0022]) / 0.01,
#         torch.tensor([-2.2857e-04, -2.1294e-04, -3.0523e-04,  2.4889e-04,  1.2937e-04,  3.4960e-04, -4.3830e-05]) / 0.0005,
#         torch.tensor([-3.7809e-05, -3.2926e-05, -2.5110e-05, 3.6659e-05, -4.7529e-05,  1.5444e-05, 2.8660e-05]) / 0.00001,
#         torch.tensor([2.1853e-05, -3.2397e-05, -1.0472e-05, 3.7024e-05, -3.8597e-04,  -3.7812e-04, 1.4447e-04]) / 0.00005,
#         torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0]) / 1,
#         torch.tensor([-100., -100., -100., -100., -100., -100., 100.]) / 1,
#     ], dim=0)))

# 冻结模型参数
if args.freeze_encoder:
    model.freeze_encoder()
if not args.freeze_weights and args.temperature > 0:
    model.model.encoder.multi_weights.requires_grad = True

if args.method == "lwf" or args.method == "derpp":
    if args.weighted_sum_layer:
        old_model = WhisperForConditionalGenerationWithWeightedSumLayer.from_pretrained(base_model)
        old_model.config.weighted_sum_layer = True
        old_model.model.encoder.weighted_sum_layer = True
        old_model.model.encoder.temperature = args.temperature
    else:
        old_model = WhisperForConditionalGeneration.from_pretrained(base_model)

    # 训练时设置为空
    old_model.config.forced_decoder_ids = None
    old_model.config.suppress_tokens = []

    # 修改模型词表大小
    old_model.resize_token_embeddings(len(processor.tokenizer))

    # 冻结模型参数
    if args.freeze_encoder:
        old_model.freeze_encoder()
    # if not args.freeze_weights:
    #     old_model.model.encoder.multi_weights.requires_grad = True

    # 预定义RE实验
    if args.name == "DERPP-PW-WSL-1-KS-SID-ER-IC-SF-ASR-RE":
        old_model.model.encoder.multi_weights = (
            torch.nn.Parameter(torch.stack([
                torch.tensor([-0.0016, -0.0016, -0.0014, -0.0013, 0.0006, -0.0002, 0.0026]) / 0.0005,
                torch.tensor([-0.0020, -0.0016, -0.0014, 0.0016, 0.0024, 0.0001, -0.0009]) / 0.0005,
                torch.tensor([-0.0014, -0.0013, -0.0011, 0.0006, 0.0010, 0.0010, 0.0012]) / 0.0005,
                torch.tensor([-0.0012, -0.0013, -0.0012, -0.0015, -0.0013, -0.0009, 0.0030]) / 0.0005,
                torch.tensor([-0.0006, -0.0007, -0.0008, -0.0009, -0.0013, -0.0010, 0.0019]) / 0.0005,
                torch.tensor([-0.0003, -0.0003, -0.0004, -0.0004, -0.0004, -0.0002, 0.0007]) / 0.0001,
                torch.tensor([-0.0006, -0.0010, -0.0011, -0.0012, -0.0011, -0.0007, 0.0024]) / 0.0005,
            ], dim=0)))

output_dir = os.path.join(args.output_dir, "whisper-base", args.name)

num_train_epochs = args.num_train_epochs
if args.task == "SID" and num_train_epochs == 20:
    num_train_epochs = 40
if args.task == "ER" and num_train_epochs == 20:
    num_train_epochs = 60
if args.name.split("-")[-1] == "SID" and num_train_epochs == 20:
    num_train_epochs = 40
if args.name.split("-")[-1] == "ER" and num_train_epochs == 20:
    num_train_epochs = 60

# 指定warmup为总训练的时间的10%
warmup_steps = int((len(train_dataset) * args.num_train_epochs / args.batch_size) * 0.1)
# breakpoint()

metric_for_best_model = {
    "RE": "loss",
    "EE": "loss",
    "KS": "acc",
    "SID": "acc",
    "ER": "acc",
    "IC": "acc",
    "SF": "slot_type_f1",
    "transcribe": "wer",
}

# 定义训练参数
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # 保存检查点和意志的目录
                             per_device_train_batch_size=args.batch_size,  # 训练batch_size大小
                             per_device_eval_batch_size=args.batch_size,  # 评估batch_size大小
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # 训练梯度累计步数
                             learning_rate=args.learning_rate,  # 学习率大小
                             warmup_steps=warmup_steps,  # 预热步数
                             num_train_epochs=num_train_epochs,  # 微调训练轮数
                             save_strategy="steps",  # 指定按照步数保存检查点
                             evaluation_strategy="steps",  # 指定按照步数评估模型
                             load_best_model_at_end=True,  # 指定是否在结束时加载最优模型
                             fp16=args.fp16,  # 是否使用半精度训练
                             report_to=["tensorboard"],  # 指定使用tensorboard保存log
                             save_steps=args.save_steps,  # 指定保存检查点的步数
                             eval_steps=args.eval_steps,  # 指定评估模型的步数
                             save_total_limit=1,  # 只保存最新检查点的数量
                             optim='adamw_torch',  # 指定优化方法
                             dataloader_num_workers=args.num_workers,  # 设置读取数据的线程数量
                             logging_steps=args.logging_steps,  # 指定打印log的步数
                             remove_unused_columns=False,  # 删除模型不需要的数据列
                             label_names=["labels"],  # 与标签对应的输入字典中的键列表
                             seed=42,  # 随机数

                             # metric_for_best_model=metric_for_best_model.get(args.task, "loss"),
                             )

# 定义训练器
if args.method == "none":
    trainer = Seq2SeqTrainer(args=training_args,
                             model=model,
                             train_dataset=train_dataset,
                             eval_dataset=valid_dataset,
                             data_collator=data_collator,
                             tokenizer=tokenizer,

                             # compute_metrics=compute_metrics(task=args.task),
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                             )

# 实例化自定义的 LwFSeq2SeqTrainer
if args.method == "lwf":
    trainer = LwFSeq2SeqTrainer(
        model=model,
        args=training_args,  # 定义你的训练参数
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        old_model=old_model,  # 传入旧任务的模型副本
        alpha=0.5,            # LwF 的权重
        temperature=2.0,       # 知识蒸馏的温度参数

        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    )

# 实例化自定义的 DERPPSeq2SeqTrainer
if args.method == "derpp":
    trainer = DERPPSeq2SeqTrainer(
        model=model,
        args=training_args,  # 定义你的训练参数
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        old_model=old_model,  # 传入旧任务的模型副本
        alpha=1.0,
        beta=1.0,
        batch_size=args.batch_size,
        memory_dataset=memory_dataset,

        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    )

model.config.use_cache = False

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存最后的模型
trainer.save_state()
# model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
