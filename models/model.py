from torch import nn, Tensor, Sequential
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch

class LLM(nn.Module):
    def __init__(self, language = "en", model_name = None):
        super().__init__()
        if model_name is None:
            raise ValueError("model_name is required")
        if language not in model_name.keys():
            raise ValueError("language is not supported")
        
        if model_name in ['llama-3.3-8b-instruct', 'llama-3.3-70b-instruct']:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_name in [ 'google/flan-t5-large', 'google/flan-t5-small']:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)    

        self.feat_size = self.model.config.hidden_size
        
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs.last_hidden_state
    
    
class MyEmo(nn.Module):
    def __init__(self, output_dropout = 0.1, model_name = None, joint_loss = 'joint', alpha = 0.2):
        super(MyEmo, self).__init__()
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
        inputs_ids, labels, label_idxs = batch
        labels = labels.to(self.device)

        # get the embeddings
        last_hidden_state = self.encoder(inputs_ids)
        
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)
        
        # compute the loss
        if self.joint_loss == 'joint':
            cel_loss = self.corr_loss(logits, labels)
            bin_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = ((1-self.alpha) * bin_loss) +(self.alpha * cel_loss) 
        elif self.joint_loss == 'cross_entropy':
            loss = F.binary_cross_entropy_with_logits(logits, label s)
        elif self.joint_loss == 'correlation':
            loss = self.corr_loss(logits, labels)
        y_pred = self.compute_predictions(logits)
        
        return loss, num_rows, y_pred, labels.cpu().numpy()
        

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
