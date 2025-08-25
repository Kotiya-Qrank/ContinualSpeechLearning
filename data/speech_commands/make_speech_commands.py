import json
import os

# 指定目录的路径
directory_path = "speech_commands_v0.01"

# 使用os.walk遍历目录及其子目录
all_files = []
for root, dirs, files in os.walk(directory_path):
    if root == "speech_commands_v0.01":
        continue
    for file_name in files:
        if root.split("/")[1] == "_background_noise_":
            continue
        # 输出每个最终文件的路径
        file_path = os.path.join(root.split("/")[1], file_name)
        all_files.append(file_path)

audio_root: str = "/data1/wgt/TransformersWhisper/data/speech_commands/speech_commands_v0.01/"

with open("speech_commands_v0.01/validation_list.txt", "r") as f:
    valid_files = f.readlines()
    valid_files = [line.rstrip('\n') for line in valid_files]

with open("speech_commands_v0.01/testing_list.txt", "r") as f:
    test_files = f.readlines()
    test_files = [line.rstrip('\n') for line in test_files]

print(len(all_files), len(valid_files), len(test_files))

train_files = []
for file in all_files:
    if file not in valid_files and file not in test_files:
        train_files.append(file)

print(len(train_files))

for i, files in enumerate([train_files, valid_files, test_files]):

    save = []
    for file in files:
        path = file
        label = file.split("/")[0]
        save_line = {
            "audio": {
                "path": audio_root + path
            },
            "sentence": label,
            "language": "English"
        }
        save.append(save_line)

    if i == 0:
        dataset = "train"
    elif i == 1:
        dataset = "valid"
    elif i == 2:
        dataset = "test"
    else:
        dataset = ""

    with open(dataset + ".json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(save, indent=4, ensure_ascii=False))
