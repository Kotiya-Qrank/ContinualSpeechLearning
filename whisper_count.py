import argparse
import functools
import json
import os

from whisper_utils import print_arguments, add_arguments
import metric

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("name", type=str, default="None", help="起个名字吧")
add_arg("output_dir", type=str, default="/datanfs2/wgt/TransformersWhisper/output8", help="训练保存模型的路径")
add_arg("base_model", type=str, default="whisper-base", help="Whisper的基础模型")
add_arg("task", type=str, default=None, help="任务类型")
add_arg("prompt", type=bool, default=True, help="是否添加prompt")
add_arg("suppress", type=bool, default=True, help="是否添加suppress tokens")
args = parser.parse_args()
print_arguments(args)

# with open(os.path.join(args.output_dir, args.base_model, args.name, "predict.json"), "r") as f:
with open(os.path.join(args.output_dir, args.base_model, args.name, "predict_" + args.task.lower() + "_" + str(int(args.prompt)) + str(int(args.suppress)) + ".json"), "r") as f:
    test_dataset = json.load(f)
    total = 0.0
    right = 0.0
    predicts = []
    labels = []
    for line in test_dataset:
        # predict = line["predict"].split("is")[-1].strip(".")
        # label = line["label"].split("is")[-1].strip(".")
        predict = line["predict"]
        label = line["label"]
        predicts.append(predict)
        labels.append(label)
        if predict == label:
            right += 1
        # else:
        #     print(predict, label)
        total += 1

    print("accuracy:", int(right), "/", int(total), "=", round(right / total * 100, 2), "%")

    if args.task == "SF":
        print("slot_type_f1:", metric.slot_type_f1(predicts, labels))
        print("slot_value_cer:", metric.slot_value_cer(predicts, labels))
        print("slot_value_wer:", metric.slot_value_wer(predicts, labels))
        print("slot_edit_f1_full:", metric.slot_edit_f1_full(predicts, labels))
        print("slot_edit_f1_part:", metric.slot_edit_f1_part(predicts, labels))
        print("wer:", metric.wer(predicts, labels))
        print("cer:", metric.cer(predicts, labels))

    if args.task == "transcribe":

        import editdistance

        pred_words_all = []
        target_words_all = []

        for predict in predicts:
            pred_words_all.append(predict.upper())
        for label in labels:
            target_words_all.append(label.upper())

        # pred_tokens_all = "|".join(pred_words_all.split())
        # pred_tokens_all = " ".join([i for i in pred_tokens_all])
        #
        # target_tokens_all = "|".join(target_words_all.split())
        # target_tokens_all = " ".join([i for i in target_tokens_all])
        #
        # """Computes WER and UER given the prediction and true transcriptions"""
        # unit_error_sum = 0.0
        # word_error_sum = 0.0
        # unit_length_sum = 0
        # word_length_sum = 0
        #
        # for pred_tokens, pred_words, target_tokens, target_words in zip(
        #         pred_tokens_all, pred_words_all, target_tokens_all, target_words_all
        # ):
        #     pred_tokens = pred_tokens.split()
        #     target_tokens = target_tokens.split()
        #     unit_error_sum += editdistance.eval(pred_tokens, target_tokens)
        #     unit_length_sum += len(target_tokens)
        #
        #     word_error_sum += editdistance.eval(pred_words, target_words)
        #     word_length_sum += len(target_words)
        #
        # uer, wer = 100.0, 100.0
        # if unit_length_sum > 0:
        #     uer = 100.0 * unit_error_sum / unit_length_sum
        # if word_length_sum > 0:
        #     wer = 100.0 * word_error_sum / word_length_sum
        #
        # print("uer:", uer)
        # print("wer:", wer)

        import jiwer
        # from whisper.normalizers import EnglishTextNormalizer
        #
        # normalizer = EnglishTextNormalizer()
        wer = jiwer.wer(pred_words_all, target_words_all)

        print(f"WER: {wer * 100:.2f} %")

    if args.task == "RE":
        gt = dict()
        pred = dict()
        gt_entities = []
        gt_relations = []
        gt_entities_relations = []
        pred_entities = []
        pred_relations = []
        pred_entities_relations = []

        for i in range(len(predicts)):

            gt_triplets = metric.parse_triplets(labels[i])
            gt[i] = gt_triplets

            pred_triplets = metric.parse_triplets(predicts[i])
            pred[i] = pred_triplets

        for i in sorted(gt):
            temp = set()
            for j in gt[i]:
                temp.add(j["head"])
                temp.add(j["tail"])
            gt_entities.append(list(temp))
        for i in sorted(pred):
            temp = set()
            for j in pred[i]:
                temp.add(j["head"])
                temp.add(j["tail"])
            pred_entities.append(list(temp))
        for i in sorted(gt):
            temp = []
            temp2 = []
            for j in gt[i]:
                rel_only = tuple([0, j["type"], 0])
                ent_rel = tuple(j.values())
                temp.append(rel_only)
                temp2.append(ent_rel)
            gt_relations.append(temp)
            gt_entities_relations.append(temp2)
        for i in sorted(pred):
            temp = []
            temp2 = []
            for j in pred[i]:
                rel_only = tuple([0, j["type"], 0])
                ent_rel = tuple(j.values())
                temp.append(rel_only)
                temp2.append(ent_rel)
            pred_relations.append(temp)
            pred_entities_relations.append(temp2)
        # print(gt_entities)
        # print(pred_entities)
        # print(gt_relations)
        # print(pred_relations)
        # print(gt_entities_relations)
        # print(pred_entities_relations)

        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        metric.entity_score(gt_entities, pred_entities)

        print("")
        print("--- Relations (Relation classification (RC)) ---")
        print("A relation is considered correct if the relation type "
              "is predicted correctly (entity type is not considered)")
        print("")
        metric.relation_score(gt_relations, pred_relations)

        print("")
        print("--- Triplets (Entities and relations (EAR)) ---")
        print("A relation is considered correct if the relation type "
              "and the entities are predicted correctly (entity type is not considered)")
        print("")
        metric.relation_score(gt_entities_relations, pred_entities_relations)
