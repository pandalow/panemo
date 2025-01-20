from torch import nn, Tensor, Sequential
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLM(nn.Module):
    def __init__(self, language = "en", model_name = None):
        super().__init__()
        if model_name is None:
            raise ValueError("model_name is required")
        
        if language not in model_name.keys():
            raise ValueError("language is not supported")
        
        if language == "en":
            self.model = AutoModelForCausalLM.from_pretrained(model_name["en"])
            self.tokenizer = AutoTokenizer.from_pretrained(model_name["en"])
        elif language == "ch":
            self.model = AutoModelForCausalLM.from_pretrained(model_name["ch"])
            self.tokenizer = AutoTokenizer.from_pretrained(model_name["ch"])
        elif language == "de":
            self.model = AutoModelForCausalLM.from_pretrained(model_name["de"])
            self.tokenizer = AutoTokenizer.from_pretrained(model_name["de"])
            
        self.feat_size = self.model.config.hidden_size
        self.n_embed = self.feat_size

    def encode(self, x, max_length = 512):
        
        if isinstance(x, str):
            x  = [x]
        elif isinstance(x, list):
            x = [item for item in x if item is not None]
        else:
            raise ValueError("x must be a string or a list of strings")
        
        encoding = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        return encoding["input_ids"]
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs.last_hidden_state
    
    
class SpanEmo(nn.Module):
    def __init__(self, output_dropout = 0.1, model_name = None, joint_loss = 'joint', alpha = 0.2):
        super(SpanEmo, self).__init__()
        self.encoder = LLM(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.joint_loss = joint_loss
        self.alpha = alpha
            
        self.ffn = nn.Sequential(
            nn.Linear(self.encoder.feat_size, self.encoder.feat_size),
            nn.ReLU(),
            nn.Dropout(n=output_dropout),
            nn.Linear(self.encoder.feat_size, 1)
        )

    def forward(self, batch):
        # prepare the input
        inputs, targets,length,label_idxs = batch
        inputs, num_rows = inputs.to(self.device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(self.device), targets.float().to(self.device)

        # encode the input
        encoded_input = self.encoder.encode(inputs.cpu().tolist())  # 确保 inputs 转换为字符串格式
        encoded_input = encoded_input.to(self.device)

        # get the embeddings
        last_hidden_state = self.encoder(encoded_input)
        
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)

        # compute the loss
        if self.joint_loss == 'joint':
            cel_loss = self.corr_loss(logits, targets)
            bin_loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss = ((1-self.alpha) * bin_loss) +(self.alpha * cel_loss) 
        elif self.joint_loss == 'cross_entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        elif self.joint_loss == 'correlation':
            loss = self.corr_loss(logits, targets)
        y_pred = self.compute_predictions(logits)
        
        return loss, num_rows, y_pred, targets.cpu().numpy()
        

    @staticmethod
    def corr_loss(y_hat, y_true, reduction='mean'):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == 'mean' else loss.sum()
    
    
    @staticmethod
    def compute_predictions(logits,threshold = 0.5):
        y_pred = torch.sigmoid(logits)
        y_pred = (y_pred > threshold).float()
        return y_pred.cpu().numpy()
