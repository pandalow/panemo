import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import MyEmoSubword
from dataset.dataloader import EmoDataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch"""

    def __init__(self, filename, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cur_date = filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'models/' + self.cur_date + '_checkpoint.pt')
        self.val_loss_min = val_loss



def evaluate_macro_f1(model, dataset, batch_size=4, device="cuda"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids, gold_labels, label_positions = [x.to(device) for x in batch]

            outputs = model((input_ids, gold_labels, label_positions))
            if isinstance(outputs, tuple): 
                _, _, logits, _ = outputs
            else: 
                logits = outputs

            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, device=device)

            preds = model.compute_predictions(logits) 

            gold_labels = gold_labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(gold_labels)

    # 拼接所有 batch
    all_preds = np.concatenate(all_preds, axis=0)   # shape [N, num_labels]
    all_labels = np.concatenate(all_labels, axis=0) # shape [N, num_labels]

    # 计算 Macro-F1 分数
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1

def train(args):

    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    device = args['device']

    # This value determines how many small batches we accumulate before performing an optimization step
    gradient_accumulation_steps = args['gradient_accumulation_steps']

    train_dataset = EmoDataset(args)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MyEmoSubword(args)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    model.train()
    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        running_loss = 0.0 
        
        for step, batch in enumerate(progress_bar):
            input_ids, labels, label_positions = [x.to(device) for x in batch]

            loss, bsz, preds, golds = model((input_ids, labels, label_positions))
            
            loss = loss / gradient_accumulation_steps
            

            loss.backward()
            
            running_loss += loss.item()  
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = running_loss 
                progress_bar.set_postfix({"loss": avg_loss})
                running_loss = 0.0

    return model