import numpy as np
import json
import random

random.seed(42)
buffer_size = 1000

with open("../speech_commands/train.json")as f:
    speech_commands_train = json.load(f)
    for line in speech_commands_train:
        line["sentence"] = "<|en|><|KS|><|notimestamps|>" + line["sentence"]

with open("../VoxCeleb1_top10/train.json")as f:
    VoxCeleb1_top10_train = json.load(f)
    for line in VoxCeleb1_top10_train:
        line["sentence"] = "<|en|><|SID|><|notimestamps|>" + line["sentence"]

with open("../IEMOCAP/train.json")as f:
    IEMOCAP_train = json.load(f)
    for line in IEMOCAP_train:
        line["sentence"] = "<|en|><|ER|><|notimestamps|>" + line["sentence"]

with open("../fluent/train.json")as f:
    fluent_train = json.load(f)
    for line in fluent_train:
        line["sentence"] = "<|en|><|IC|><|notimestamps|>" + line["sentence"]

with open("../SNIPS/train.json")as f:
    SNIPS_train = json.load(f)
    for line in SNIPS_train:
        line["sentence"] = "<|en|><|SF|><|notimestamps|>" + line["sentence"]

with open("../LibriSpeech/train.json")as f:
    LibriSpeech_train = json.load(f)
    for line in LibriSpeech_train:
        line["sentence"] = "<|en|><|transcribe|><|notimestamps|>" + line["sentence"]

speech_commands_memory = random.sample(speech_commands_train, buffer_size)
VoxCeleb1_top10_memory = random.sample(VoxCeleb1_top10_train, buffer_size)
IEMOCAP_memory = random.sample(IEMOCAP_train, buffer_size)
fluent_memory = random.sample(fluent_train, buffer_size)
SNIPS_memory = random.sample(SNIPS_train, buffer_size)
LibriSpeech_memory = random.sample(LibriSpeech_train, buffer_size)

with open("speech_commands_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(speech_commands_memory, indent=4, ensure_ascii=False))

with open("VoxCeleb1_top10_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(VoxCeleb1_top10_memory, indent=4, ensure_ascii=False))

with open("IEMOCAP_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(IEMOCAP_memory, indent=4, ensure_ascii=False))

with open("fluent_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(fluent_memory, indent=4, ensure_ascii=False))

with open("SNIPS_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(SNIPS_memory, indent=4, ensure_ascii=False))

with open("LibriSpeech_memory.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(LibriSpeech_memory, indent=4, ensure_ascii=False))
