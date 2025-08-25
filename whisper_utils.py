import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from typing import Any, List, Dict, Union

import json
import mmap

import struct

from tqdm import tqdm

from sklearn.metrics import classification_report, precision_score, \
    recall_score, f1_score, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from transformers_whisper import WhisperProcessor
import metric

import random
random.seed(42)

class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            mono: 是否将音频转换成单通道，这个必须是True
            language: 微调数据的语言
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.data_list: List[dict] = []
        self.tensor_data_list: List[tuple] = []
        self.data = []
        # 加载数据列表
        self._load_data_list()
        # 加载tensor列表
        # self._load_tensor_data_list()

    # 加载数据列表
    def _load_data_list(self):
        # 获取数据列表
        with open(self.data_list_path, 'r', encoding='utf-8') as f:
            lines = json.load(f)
        self.data_list = []
        for line in tqdm(lines, desc='读取数据列表'):
            # # 跳过超出长度限制的音频
            # if line["duration"] < self.min_duration:
            #     continue
            # if self.max_duration != -1 and line["duration"] > self.max_duration:
            #     continue
            self.data_list.append(dict(line))

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 分割读取音频
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        # 转成单通道
        if self.mono:
            sample = librosa.to_mono(sample)
        return sample, sample_rate, transcript, language

    def _load_tensor_data_list(self):
        for idx in tqdm(range(len(self.data_list)), desc='加载数据列表'):
            sample, sample_rate, transcript, language = self._get_list_data(idx)
            # 可以为单独数据设置语言
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # 获取log-Mel特征和标签ID
                data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
                self.tensor_data_list.append(data)
            else:
                continue

    def __getitem__(self, idx):
        try:
            # 从数据列表里面获取音频数据、采样率和文本
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            if transcript == "_silence_":
                random_offset = random.randint(0, len(sample) - 16000)
                sample = sample[random_offset: random_offset + 16000]
            # 可以为单独数据设置语言
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # 获取log-Mel特征和标签ID
                data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
                return data
        # try:
        #     # 从数据列表里面获取音频数据、采样率和文本
        #     data = self.tensor_data_list[idx]
        #     return data
        except Exception as e:
            print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str_none(val):
    if val == 'None':
        return None
    else:
        return val


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def whisper_infer(audio_path, processor, model, task, forced_decoder_ids, suppress_tokens, skip_special_tokens=True):
    # 读取音频
    sample, sr = librosa.load(audio_path, sr=16000)
    data = processor(audio=sample, sampling_rate=16000, text="transcript")
    duration = sample.shape[-1] / sr
    # assert duration < 30, f"本程序只适合推理小于30秒的音频，当前音频{duration}秒，请使用其他推理程序!"
    # 预处理音频
    input_features = processor(sample, sampling_rate=sr, return_tensors="pt",
                               do_normalize=False).input_features.cuda()
    # 开始识别
    predicted_ids = model.generate(input_features, task=task, forced_decoder_ids=forced_decoder_ids,
                                   suppress_tokens=suppress_tokens, max_new_tokens=256)
    # breakpoint()
    if forced_decoder_ids[1][1] == 51870:
        import torch

        # 找到所有 13 的位置
        indices_of_13 = (predicted_ids == 13).nonzero(as_tuple=True)[1]

        if len(indices_of_13) >= 1:
            first_13_index = indices_of_13[0].item()

            # 将 predicted_ids 转换为列表
            predicted_list = predicted_ids[:, :first_13_index + 1][0].tolist()

            # 生成 (位置, token_id) 的列表
            forced_decoder_ids = [(i + 1, token_id) for i, token_id in enumerate(predicted_list)]
            forced_decoder_ids.extend([(len(predicted_list) + 1, 24753),(len(predicted_list) + 2, 307),(len(predicted_list) + 4, 13)])

            # suppress_tokens = list(range(len(processor.tokenizer)))
            # for id in [1318, 5811, 3738, 6654, 2856, 13669, 12684, 8544, 5523, 6521, 6933, 17564, 3669, 4649, 13, 50257]:
            #     suppress_tokens.remove(id)

            predicted_ids = model.generate(input_features, task=task, forced_decoder_ids=forced_decoder_ids,
                                           suppress_tokens=suppress_tokens, max_new_tokens=256)
            # breakpoint()
            # 找到所有 13 的位置
            indices_of_13 = (predicted_ids == 13).nonzero(as_tuple=True)[1]

            # 取第二个 13 的位置
            if len(indices_of_13) >= 2:
                # first_13_index = indices_of_13[0].item()
                second_13_index = indices_of_13[1].item()

                # 将 predicted_ids 转换为列表
                predicted_list = predicted_ids[:, :second_13_index + 1][0].tolist()

                # 生成 (位置, token_id) 的列表
                forced_decoder_ids = [(i + 1, token_id) for i, token_id in enumerate(predicted_list)]
                forced_decoder_ids.extend([(len(predicted_list) + 1, 12859), (len(predicted_list) + 2, 399), (len(predicted_list) + 3, 307),])

                # suppress_tokens = list(range(len(processor.tokenizer)))
                # for id in [11211, 6525, 6022, 5675, 2861, 13, 50257]:
                #     suppress_tokens.remove(id)

                predicted_ids = model.generate(input_features, task=task, forced_decoder_ids=forced_decoder_ids,
                                               suppress_tokens=suppress_tokens, max_new_tokens=256)
                # breakpoint()
    # 解码结果
    result = processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_tokens)[0].strip()
    return result


# Interface
def get_dataset(dataset, mode, batch_size, num_workers):
    if mode == 'train':
        if dataset.data_list_path.split("/")[1] == "speech_commands":
            return get_balanced_dataloader(dataset, batch_size, num_workers, drop_last=True)
        else:
            return get_dataloader(dataset, batch_size, num_workers, drop_last=True)
    elif mode == 'valid':
        if dataset.data_list_path.split("/")[1] == "speech_commands":
            return get_balanced_dataloader(dataset, batch_size, num_workers, drop_last=False)
        else:
            return get_dataloader(dataset, batch_size, num_workers, drop_last=False)
    elif mode == 'test':
        return get_dataloader(dataset, batch_size, num_workers, drop_last=False)


def get_balanced_dataloader(dataset, batch_size, num_workers, drop_last=False):
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset.sample_weights))
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def get_dataloader(dataset, batch_size, num_workers, drop_last=False):
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(samples):
    return zip(*samples)


def compute_metrics(task):
    processor = WhisperProcessor.from_pretrained("pretrained_models/whisper-processor.final",
                                                 language="English",
                                                 task=None,
                                                 local_files_only=True)

    def accuracy(pred):
        predict_ids = pred.predictions[0].argmax(axis=2)
        label_ids = pred.label_ids
        predicts = processor.batch_decode(predict_ids, skip_special_tokens=True)
        labels = processor.batch_decode(label_ids, skip_special_tokens=True)
        same = np.array(labels) == np.array(predicts)
        return {
            "acc": same.sum() / len(same),
            "right": same.sum(),
            "total": len(same),
        }

    def compute_re(pred):
        pass

    def compute_ee(pred):
        pass

    def compute_sf(pred):
        predict_ids = pred.predictions[0].argmax(axis=2)
        label_ids = pred.label_ids
        predicts = processor.batch_decode(predict_ids, skip_special_tokens=True)
        labels = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "slot_type_f1": metric.slot_type_f1(predicts, labels),
            "slot_value_cer": metric.slot_value_cer(predicts, labels),
            "slot_value_wer": metric.slot_value_wer(predicts, labels),
            "slot_edit_f1_full": metric.slot_edit_f1_full(predicts, labels),
            "slot_edit_f1_part": metric.slot_edit_f1_part(predicts, labels),
            "wer": metric.wer(predicts, labels),
            "cer": metric.cer(predicts, labels),
        }

    def compute_asr(pred):
        predict_ids = pred.predictions[0]
        label_ids = pred.label_ids
        predicts = processor.batch_decode(predict_ids, skip_special_tokens=True)
        labels = processor.batch_decode(label_ids, skip_special_tokens=True)

        import editdistance
        pred_words_all = predicts.upper()
        target_words_all = labels.upper()

        pred_tokens_all = "|".join(pred_words_all.split())
        pred_tokens_all = " ".join([i for i in pred_tokens_all])

        target_tokens_all = "|".join(target_words_all.split())
        target_tokens_all = " ".join([i for i in target_tokens_all])

        """Computes WER and UER given the prediction and true transcriptions"""
        unit_error_sum = 0.0
        word_error_sum = 0.0
        unit_length_sum = 0
        word_length_sum = 0

        for pred_tokens, pred_words, target_tokens, target_words in zip(
                pred_tokens_all, pred_words_all, target_tokens_all, target_words_all
        ):
            pred_tokens = pred_tokens.split()
            target_tokens = target_tokens.split()
            unit_error_sum += editdistance.eval(pred_tokens, target_tokens)
            unit_length_sum += len(target_tokens)

            word_error_sum += editdistance.eval(pred_words, target_words)
            word_length_sum += len(target_words)

        uer, wer = 100.0, 100.0
        if unit_length_sum > 0:
            uer = 100.0 * unit_error_sum / unit_length_sum
        if word_length_sum > 0:
            wer = 100.0 * word_error_sum / word_length_sum

        return {
            "uer": uer,
            "wer": wer,
        }

    metric_to_use = {
        "RE": compute_re,
        "EE": compute_ee,
        "KS": accuracy,
        "SID": accuracy,
        "ER": accuracy,
        "IC": accuracy,
        "SF": compute_sf,
        "transcribe": compute_asr,
    }

    return metric_to_use.get(task, None)


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
