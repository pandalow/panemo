import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import bitsandbytes as bnb
from config import config


class CustomModel(nn.Module):
    def __init__(
        self,
        model_name='hfl/llama-3-chinese-8b-instruct-v3',
        num_labels=config.num_labels
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # 使用 bitsandbytes 进行 8-bit 量化
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            load_in_8bit=config.load_in_8bit,  # 8-bit 量化
            device_map="auto"
        )

        hidden_size = self.encoder.config.hidden_size
        self.fnn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, num_labels)
        ).to(self.encoder.device)  # ensure FFN is on the same device


    def forward(self, input_ids, attention_mask, labels=None):
        # quantize model forward propagation
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # pool layer
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden * attention_mask.unsqueeze(-1)
        pooled = pooled.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-8)  # avoid division by zero

        logits = self.fnn(pooled)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
                pos_weight=torch.tensor([2.0] * logits.shape[-1]).to(logits.device)
            )
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}



