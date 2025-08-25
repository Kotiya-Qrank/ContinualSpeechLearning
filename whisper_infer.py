import argparse
import functools
import json
from tqdm import tqdm
import os

from transformers import WhisperForConditionalGeneration

from transformers_whisper.modeling_whisper import WhisperForConditionalGenerationWithWeightedSumLayer
from transformers_whisper import WhisperProcessor
from whisper_utils import print_arguments, add_arguments, whisper_infer

# 设置 cpu 数量
# cpu_cores0 = [0, 1, 2, 3, 4, 5, 6, 7]
cpu_cores1 = [8, 9, 10, 11, 12, 13, 14, 15]
cpu_cores2 = [16, 17, 18, 19, 20, 21, 22, 23]
cpu_cores3 = [24, 25, 26, 27, 28, 29, 30, 31]
cpu_cores4 = [32, 33, 34, 35, 36, 37, 38, 39]
cpu_cores5 = [40, 41, 42, 43, 44, 45, 46, 47]
cpu_cores6 = [48, 49, 50, 51, 52, 53, 54, 55]
cpu_cores = []
# cpu_cores.extend(cpu_cores0)
cpu_cores.extend(cpu_cores1)
cpu_cores.extend(cpu_cores2)
cpu_cores.extend(cpu_cores3)
cpu_cores.extend(cpu_cores4)
cpu_cores.extend(cpu_cores5)
cpu_cores.extend(cpu_cores6)
# os.sched_setaffinity(0, cpu_cores)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("name", type=str, default="None", help="起个名字吧")
add_arg("test_path", type=str, default=None, help="预测的音频路径或json文件")
add_arg("output_dir", type=str, default="/datanfs2/wgt/TransformersWhisper/output8", help="训练保存模型的路径")
add_arg("base_model", type=str, default="whisper-base", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("processor_path", type=str, default="whisper-processor.final", help="编码器的路径，或者是huggingface上模型的名称")
add_arg("weighted_sum_layer", type=bool, default=False, help="是否使用weighted sum layer")
add_arg("language", type=str, default="English", help="设置语言，可全称也可简写，如果为None则预测的是多语言")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("skip_special_tokens", type=bool, default=True, help="是否跳过特殊字符")
add_arg("task", type=str, default="transcribe", help="模型的任务")
add_arg("temperature", type=float, default=-1, help="softmax的温度")
add_arg("prompt", type=bool, default=True, help="是否添加prompt")
add_arg("suppress", type=bool, default=True, help="是否添加suppress tokens")

args = parser.parse_args()
print_arguments(args)

test_path = args.test_path
if not args.test_path:
    if args.task == "KS":
        test_path = "data/speech_commands/test.json"
    elif args.task == "SID":
        test_path = "data/VoxCeleb1_top10/test.json"
    elif args.task == "ER":
        test_path = "data/IEMOCAP/test.json"
    elif args.task == "IC":
        test_path = "data/fluent/test.json"
    elif args.task == "SF":
        test_path = "data/SNIPS/test.json"
    elif args.task == "transcribe":
        test_path = "data/LibriSpeech/test.json"
    elif args.task == "RE":
        test_path = "data/re-tacred/test.json"

# model_path = os.path.join(args.output_dir, args.base_model, args.name)
model_path = os.path.join(args.output_dir, "whisper-base", args.name)

# 获取Whisper的特征提取器、编码器和解码器
processor = WhisperProcessor.from_pretrained(os.path.join("pretrained_models", args.processor_path),
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
if args.prompt:
    prompt = [50258, 50259, forced_decoder_ids[1][1], 50363]
    if args.task == "KS":
        if "_nothe" in args.name:
            prompt = [39206, 474, 307]
        elif "_the" in args.name:
            prompt = [2278, 5622, 307]
        else:
            prompt = "x"

    if args.task == "SID":
        if "_origin" in args.name:
            prompt = "x."
        elif "_nothe" in args.name:
            prompt = "Speaker is xx."
        else:
            prompt = "Speaker is xx."

    if args.task == "ER":
        if "_origin" in args.name:
            prompt = "x."
        elif "_the" in args.name:
            prompt = "The emotion is x."
        else:
            prompt = "Emotion is x."

    if args.task == "IC":
        prompt = "x"
        if "_origin" in args.name:
            prompt += []
        elif "_the" in args.name:
            prompt += [2278, 3069, 307]
        else:
            prompt = "Action is"
        # 2278, 3069, 307     The action is

    if args.task == "SF":
        prompt = ""

    if args.task == "transcribe":
        prompt = ""

    if args.task == "RE":
        prompt = "Subject is"

    input_ids = processor.tokenizer(prompt)["input_ids"]
    forced_decoder_ids = [(i, input_ids[i]) for i in range(1, len(input_ids)) if input_ids[i] not in [2031, 87, 30569]]
    if args.task == "KS":
        # forced_decoder_ids = [(1, 50259), (2, 51867), (3, 50363), (5, 50257)]
        forced_decoder_ids = [(1, 50259), (2, 51867), (3, 50363)]
    if args.task == "SID":
        forced_decoder_ids = [(1, 50259), (2, 51868), (3, 50363), (4, 19588), (5, 4003), (6, 307), (7,220), (9,13), (10,50257)]
    if args.task == "IC":
        forced_decoder_ids = [(1, 50259), (2, 51870), (3, 50363), (4, 32), (5, 882), (6, 307)]
    if args.task == "SF":
        forced_decoder_ids = [(1, 50259), (2, 51871), (3, 50363)]
    if args.task == "transcribe":
        forced_decoder_ids = [(1, 50259), (2, 50359), (3, 50363)]
    if args.task == "RE":
        forced_decoder_ids = [(1, 50259), (2, 51865), (3, 50363), (4, 39582), (5, 1020), (6, 307)]

suppress_tokens = []
if args.suppress:
    remove_list = []
    if args.task == "KS":
        # remove_list += [766, 1590, 26161, 30605, 655, 62, 26161, 3197, 77, 648, 62, 572, 1411, 558, 760, 2086, 322, 493, 352]
        # "off stop no left right down yes on up go"
        remove_list += [4506, 13559, 1771, 41761, 1938, 5093, 2346, 266, 1010, 1571]
        remove_list += [50257]
        # remove_list += [51865, 51866]
        # remove_list += [53200, 53201]
        # remove_list += [39206, 474, 5622, 307]
        remove_list += [30605, 655, 3197, 77, 648, ]
        remove_list += [62]

    if args.task == "SID":
        remove_list += [51890, 52580, 52981 , 52856, 52267, 52805, 51966, 52815, 52073, 53081]
        if "_origin" in args.name:
            pass

        elif "_nothe" in args.name:
            if not args.prompt:
                remove_list += [19588, 4003, 307, 13]

        else:
            if not args.prompt:
                remove_list += [2278, 8145, 307, 13]

        # "Speaker speaker <|speaker19|> <|speaker709|> <|speaker1110|> <|speaker985|> <|speaker396|> <|speaker934|> <|speaker95|> <|speaker944|> <|speaker202|> <|speaker1210|>"


    if args.task == "ER":
        remove_list += [2055, 10598, 6884, 4227]
        if "_origin" in args.name:
            pass

        elif "_the" in args.name:
            if not args.prompt:
                remove_list += [22285, 19228, 307, 13]

        else:
            if not args.prompt:
                remove_list += [2278, 8913, 307, 13]

        # "Emotion happy neutral angry sad is emotion"
        # remove_list += [17888, 716, 43574, 656, 627, 82, 345,]

    if args.task == "IC":
        # remove_list = range(51871, 51902)
        # remove_list += [32, 882, 6525, 1318, 5811, 11211, 13615, 1565, 3738, 45428, 473, 5675, 2861, 6654, 24753, 6022, 2856, 13669, 12684, 1319, 8544, 5523, 307, 3488, 11514, 6521, 6933, 12859, 399, 17564, 3669, 4649]
        # "Action kitchen music lights bedroom activate bring heat deactivate washroom shoes Object none language newspaper lamp change juice volume is increase decrease German Korean Location socks English Chinese"
        # remove_list += [23397, 473, 1443, 278,1479, 23397, 473, 15431, 4647, 265, 651, 1479, 14066, 651]
        # remove_list += [3069, 2657, 4914,]

        #  activate deactivate bring change language increase decrease
        remove_list += [13615, 45428, 473, 1565, 1319, 2856, 3488, 11514]

        # #  music lights heat shoes language newspaper lamp juice volume German Korean socks English Chinese
        remove_list += [1318, 5811, 3738, 6654, 2856, 13669, 12684, 8544, 5523, 6521, 6933, 17564, 3669, 4649,]
        #
        # #  bedroom kitchen none washroom
        remove_list += [11211, 6525, 6022, 5675, 2861]

        remove_list += [13]
        remove_list += [50257]

    if args.task == "SF":
        remove_list += range(50257)
        remove_list += [50257]
        remove_list += range(53122, 53200)

    if args.task == "transcribe":
        remove_list += range(50257)
        remove_list += [50257]

    if args.task == "RE":
        remove_list += range(50257)
        remove_list += [50257]

    suppress_tokens = list(range(len(processor.tokenizer)))
    # suppress_tokens.remove(50257)
    # suppress_tokens.remove(50363)
    # suppress_tokens.remove(11)  # ","
    # suppress_tokens.remove(13)  # "."
    # suppress_tokens.remove(440)  # " The"
    # suppress_tokens.remove(2278)  # "The"
    remove_set = set(remove_list)
    for id in remove_set:
        suppress_tokens.remove(id)


# 获取最优模型
# best_model_checkpoint = "output/whisper-base/KS-SID-ER_replay2/checkpoint-final"
# best_model_checkpoint = "pretrained_models/whisper-base"
with open(model_path + "/trainer_state.json", "r")as f:
    trainer_state = json.load(f)
    best_model_checkpoint = trainer_state["best_model_checkpoint"]

# best_model_checkpoint = "/data1/wgt/TransformersWhisper/pretrained_models/whisper-base"
# 获取模型
if args.temperature > 0: args.weighted_sum_layer = True
if args.weighted_sum_layer:
    model = WhisperForConditionalGenerationWithWeightedSumLayer.from_pretrained(
        best_model_checkpoint,
        device_map="auto",
        local_files_only=args.local_files_only
    )
    model.model.encoder.temperature = args.temperature
else:
    model = WhisperForConditionalGeneration.from_pretrained(
        best_model_checkpoint,
        device_map="auto",
        local_files_only=args.local_files_only
    )

# processor = WhisperProcessor.from_pretrained(os.path.join("pretrained_models/whisper-base"),
#                                              language=args.language,
#                                              task=args.task,
#                                              local_files_only=args.local_files_only)
# model = WhisperForConditionalGeneration.from_pretrained(
#         "pretrained_models/whisper-base",
#         device_map="auto",
#         local_files_only=args.local_files_only
#     )
# model.resize_token_embeddings(len(processor.tokenizer))
# model = WhisperForConditionalGeneration.from_pretrained(
#         "output2/whisper-base/SID_7/checkpoint-6000",
#         device_map="auto",
#         local_files_only=args.local_files_only
#     )
model.eval()
# breakpoint()

if args.temperature > 0:
    import torch
    multi_weights = model.model.encoder.multi_weights
    norm_multi_weights = torch.nn.functional.softmax(multi_weights / args.temperature, dim=-1)
    print(multi_weights)
    print(norm_multi_weights)
    # model.model.encoder.multi_weights = (
    #     torch.nn.Parameter(torch.stack([
    #         torch.tensor([-3.5991e-04, -2.1107e-04, -1.4693e-04, -5.3523e-05, -2.1837e-04, 2.4982e-04, 5.8946e-04]) / 0.0005,
    #         torch.tensor([-0.0011, -0.0003, -0.0004, 0.0006, 0.0008, 0.0005, -0.0003]) / 0.0005,
    #         torch.tensor([-0.0010, -0.0005, -0.0003, 0.0006, 0.0006, 0.0003, 0.0005]) / 0.0005,
    #         torch.tensor([-0.0012, -0.0008, -0.0007, -0.0011, -0.0009, -0.0011, 0.0020]) / 0.0005,
    #         torch.tensor([-0.0005, -0.0005, -0.0007, -0.0006, -0.0015, -0.0011, 0.0016]) / 0.0005,
    #         torch.tensor([-0.0007, -0.0011, -0.0012, -0.0012, -0.0012, -0.0007, 0.0021]) / 0.0005,
    #         torch.tensor([0., 0., 0., 0., 0., 0., 0.]) / 1,
    #     ], dim=0)))
    # breakpoint()
    # exit(0)

task = args.task if args.weighted_sum_layer else None

if test_path.endswith(".wav"):
    result = whisper_infer(test_path, processor, model, task, forced_decoder_ids, suppress_tokens, skip_special_tokens=args.skip_special_tokens)
    print(f"识别结果：{result}")

elif test_path.endswith(".json"):
    with open(test_path, "r") as f:
        test_dataset = json.load(f)
        save = []
        total = 0.0
        right = 0.0
        for line in tqdm(test_dataset):
            audio_path = line["audio"]["path"]
            result = whisper_infer(audio_path, processor, model, task, forced_decoder_ids, suppress_tokens)
            label = line["sentence"]
            if result == label:
                right += 1
            total += 1
            save.append({
                    "audio_path": audio_path,
                    "label": label,
                    "predict": result
            })
        print("precision:", int(right), "/", int(total), "=", round(right / total * 100, 2), "%")

    with open(os.path.join(model_path, "predict_" + args.task.lower() + "_" + str(int(args.prompt)) + str(int(args.suppress)) + ".json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(save, indent=4, ensure_ascii=False))
