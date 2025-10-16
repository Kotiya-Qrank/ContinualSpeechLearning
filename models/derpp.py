import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
import numpy as np


class DERPPSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, old_model=None, alpha=1.0, beta=1.0, batch_size=16, memory_dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_model = old_model.cuda()
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.memory_dataset = memory_dataset

    def get_data(self):
        choice = np.random.choice(len(self.memory_dataset), self.batch_size, replace=False)
        buf_inputs = self.data_collator([self.memory_dataset[i] for i in choice]).to("cuda")
        return buf_inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取当前模型的输出
        outputs = model(**inputs)
        loss = outputs.loss

        buf_inputs = self.get_data()
        buf_logits = self.old_model(**buf_inputs).logits
        buf_labels = buf_inputs["labels"]

        buf_outputs = self.model(**buf_inputs).logits
        loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)
        loss += self.beta * F.cross_entropy(buf_outputs.view(-1, buf_outputs.size(-1)), buf_labels.view(-1))

        return (loss, outputs) if return_outputs else loss
