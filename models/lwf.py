import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))

class LwFSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, old_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_model = old_model.cuda()
        self.alpha = alpha
        self.temperature = temperature
        self.soft = torch.nn.Softmax(dim=1)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取当前模型的输出
        outputs = model(**inputs)
        loss = outputs.loss

        # 如果有旧模型存在，计算知识蒸馏损失
        if self.old_model is not None:
            with torch.no_grad():
                old_outputs = self.old_model(**inputs)

            # 获取新旧模型的 logits
            logits = outputs.logits
            old_logits = old_outputs.logits

            # 计算知识蒸馏损失
            # distill_loss = F.kl_div(
            #     F.log_softmax(logits / self.temperature, dim=-1),
            #     F.softmax(old_logits / self.temperature, dim=-1),
            #     reduction="batchmean"
            # ) * (self.temperature ** 2)

            distill_loss = modified_kl_div(smooth(self.soft(old_logits), 2, 1),
                            smooth(self.soft(logits), 2, 1))

            # 将知识蒸馏损失加到总损失上
            loss += self.alpha * distill_loss

        return (loss, outputs) if return_outputs else loss
