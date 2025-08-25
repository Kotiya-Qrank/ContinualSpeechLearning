import json
import matplotlib.pyplot as plt

for dataset in ["speech_commands", "VoxCeleb1_top10", "IEMOCAP", "fluent"]:
    data_count = dict()
    with open(dataset + "_memory.json")as f:
        train = json.load(f)
        for line in train:
            label = line["sentence"]
            if label in data_count.keys():
                data_count[label] += 1
            else:
                data_count[label] = 1
    print(data_count)

    # 提取键和值
    categories = list(data_count.keys())
    values = list(data_count.values())

    # 创建柱状图
    plt.bar(categories, values, color='skyblue')

    # 添加标题和标签
    plt.title('Categories by Value')
    plt.xlabel('Categories')
    plt.ylabel('Values')

    # 显示图形
    plt.show()
