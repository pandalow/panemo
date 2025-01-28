from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from utils import pool_subwords

class MyEmoSubword(nn.Module):
    """
    多头注意力 + 子词池化 + 残差 MLP:
    - 兼容 'causal' / 'seq2seq'
    - 可做多标签情感分类 (joint_loss / corr_loss / BCE)
    """
    def __init__(self, args):
        """
        :param args: dict, 需包含:
          - model_type: "causal" or "seq2seq"
          - model_path: str, HF模型名称/路径 (支持多语言)
          - num_labels: int, 标签数
          - dropout: float
          - joint_loss: str, 'joint' / 'cross_entropy' / 'correlation' / ...
          - alpha: float, 在 'joint' loss 中对应 corr_loss 的权重
          - pool_type: "mean" or "max"
          - num_heads: int, 多头注意力的头数
        """
        super().__init__()
        self.args = args

        # 1) Load HF model
        model_type = args.get('model_type', 'seq2seq')
        model_path = args.get('model_path', 'google/mt5-base')  # 更换为多语言模型
        self.pool_type = args.get('pool_type', 'mean')
        self.joint_loss = args.get('joint_loss', 'joint')
        self.alpha = args.get('alpha', 0.2)

        if model_type == 'causal':
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.hidden_size = self.model.config.hidden_size
        elif model_type == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            # 对于 mT5 / MBART，这里一般是 d_model
            if hasattr(self.model.config, 'd_model'):
                self.hidden_size = self.model.config.d_model
            else:
                self.hidden_size = self.model.config.hidden_size
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model_type = model_type

        dropout = args.get('dropout', 0.1)
        num_labels = args['num_labels']
        num_heads = args.get('num_heads', 8)

        # ========== 多头注意力 + 残差 + LayerNorm ==========
        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_mha = nn.LayerNorm(self.hidden_size)

        # ========== Bottleneck FFN + 残差 + LayerNorm ==========
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Dropout(dropout)
        )
        self.ln_ffn = nn.LayerNorm(self.hidden_size)

        # ========== 分类器 ==========
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, batch):
        """
        batch: (input_ids, labels, label_positions)
        return: (loss, batch_size, preds, gold_labels)
        """
        if len(batch) != 3:
            raise ValueError("Expect 3 elements: (input_ids, labels, label_positions)")
        input_ids, labels, label_positions = batch

        device = input_ids.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        label_positions = label_positions.to(device)

        # 构造 attention_mask
        pad_token_id = getattr(self.model.config, 'pad_token_id', None)
        if pad_token_id is not None and pad_token_id >= 0:
            attention_mask = (input_ids != pad_token_id).long()
        else:
            attention_mask = None

        # 1) 取 hidden states
        if self.model_type == 'causal':
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        else:
            # seq2seq 模型一般都带有 encoder
            enc_out = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = enc_out.last_hidden_state  # [batch, seq_len, hidden_size]

        batch_size, seq_len, hid_dim = last_hidden_state.shape
        num_labels = labels.size(1)

        # 2) 子词池化 => [batch_size, num_labels, hidden_size]
        label_hidden = torch.zeros(batch_size, num_labels, hid_dim, device=device)
        for b in range(batch_size):
            hs = last_hidden_state[b]  # [seq_len, hidden_size]
            pos = label_positions[b]   # [num_labels, 2]
            for i in range(num_labels):
                start_pos, length = pos[i].tolist()
                pooled = pool_subwords(hs, start_pos, length, pool_type=self.pool_type)
                label_hidden[b, i] = pooled

        # A) 多头注意力
        attn_output, _ = self.mha(
            query=label_hidden, 
            key=last_hidden_state,  # 原始文本向量作为 Key
            value=last_hidden_state
            )

        # 残差 + LayerNorm
        attn_output = self.ln_mha(attn_output + label_hidden)

        # B) Bottleneck FFN
        bsz, n_labels, hid_dim = attn_output.shape
        out_2d = attn_output.reshape(bsz * n_labels, hid_dim)

        z2 = self.ffn(out_2d)  
        z2 = z2 + out_2d               # 残差
        z2 = self.ln_ffn(z2)           # LayerNorm

        out_final = z2.view(bsz, n_labels, hid_dim)

        # 3) 分类器 => logits
        logits = self.classifier(out_final).squeeze(-1)  # [batch_size, num_labels]

        # 4) 计算多标签loss
        if self.joint_loss == 'joint':
            cel_loss = self.corr_loss(logits, labels)
            bin_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = ((1 - self.alpha) * bin_loss) + (self.alpha * cel_loss)
        elif self.joint_loss == 'cross_entropy':
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        elif self.joint_loss == 'correlation':
            loss = self.corr_loss(logits, labels)
        else:
            # 默认 BCE
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 5) 预测
        preds = self.compute_predictions(logits)
        return loss, batch_size, preds, labels.cpu().numpy()

    @staticmethod
    def corr_loss(y_hat, y_true):
        pos_logits = y_hat[y_true == 1]
        neg_logits = y_hat[y_true == 0]
        
        if pos_logits.shape[0] == 0 or neg_logits.shape[0] == 0:
            return torch.tensor(0.0, device=y_hat.device)
            
        pos = pos_logits.unsqueeze(1)  # [P, 1]
        neg = neg_logits.unsqueeze(0)  # [1, N]
        
        # 使用margin-based loss
        return F.relu(neg - pos + 0.3).mean()

    @staticmethod
    def compute_predictions(logits: Tensor, threshold=0.5):
        """
        logits: [batch_size, num_labels]
        return: 0/1 numpy array
        """
        sigm = torch.sigmoid(logits)
        preds = (sigm > threshold).float()
        return preds.cpu().numpy()