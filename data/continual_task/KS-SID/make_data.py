import numpy as np
import json
import random

random.seed(42)
buffer_size = 1000
batch_size = 16

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

with open("../speech_commands_memory.json")as f:
    speech_commands_memory = json.load(f)

with open("../VoxCeleb1_top10_memory.json")as f:
    VoxCeleb1_top10_memory = json.load(f)

with open("../IEMOCAP_memory.json")as f:
    IEMOCAP_memory = json.load(f)

with open("../fluent_memory.json")as f:
    fluent_memory = json.load(f)

with open("../SNIPS_memory.json")as f:
    SNIPS_memory = json.load(f)

with open("../LibriSpeech_memory.json")as f:
    LibriSpeech_memory = json.load(f)

train = (speech_commands_memory
         + VoxCeleb1_top10_memory
         + IEMOCAP_memory
         + fluent_memory
         + SNIPS_memory
         + LibriSpeech_memory)

memory = [""] * buffer_size
num_seen_examples = 0
for i in range(buffer_size * 1):
    index = reservoir(num_seen_examples, buffer_size)
    num_seen_examples += 1
    if index >= 0:
        memory[index] = train[index]

with open("../../VoxCeleb1_top10/train.json")as f:
    VoxCeleb1_top10_train = json.load(f)
    for line in VoxCeleb1_top10_train:
        line["sentence"] = "<|en|><|SID|><|notimestamps|>" + line["sentence"]
with open("../../VoxCeleb1_top10/valid.json")as f:
    VoxCeleb1_top10_valid = json.load(f)
    for line in VoxCeleb1_top10_valid:
        line["sentence"] = "<|en|><|SID|><|notimestamps|>" + line["sentence"]
with open("../../VoxCeleb1_top10/test.json")as f:
    VoxCeleb1_top10_test = json.load(f)
    for line in VoxCeleb1_top10_test:
        line["sentence"] = "<|en|><|SID|><|notimestamps|>" + line["sentence"]

half_batch_size = int(batch_size / 2)
new_train = VoxCeleb1_top10_train
batch_num = int(len(VoxCeleb1_top10_train) // half_batch_size)

for i in range(batch_num, 0, -1):
    insert_loc = i * half_batch_size
    insert_memory = random.sample(memory, half_batch_size)
    new_train = new_train[:insert_loc] + insert_memory + new_train[insert_loc:]

with open("train.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(new_train, indent=4, ensure_ascii=False))
with open("valid.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(VoxCeleb1_top10_valid, indent=4, ensure_ascii=False))
with open("test.json", 'w', encoding='utf-8') as file:
    file.write(json.dumps(VoxCeleb1_top10_test, indent=4, ensure_ascii=False))
