from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import tqdm

class EmoDataset(Dataset):
    def __init__(self, args, file_path, transform=None):
        self.args = args
        self.transform = transform
        self.file_path = file_path
        self.max_len = args.max_len
        
        train_data = pd.read_csv(file_path)
        self.data = train_data['text'].tolist()
        
        if file_path.startswith('eng_train'):
            self.labels = train_data['joy', 'sadness', 'anger', 'fear', 'suprise'].tolist()
        else:
            self.labels = train_data['joy', 'sadness', 'anger', 'fear', 'disgust', 'suprise'].tolist()
            
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.input_ids, self.input_length, self.lengths, self.label_idxs = self.process()

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        input_ids, label_idxs = 
        
        

        return 
      
    @staticmethod
    def process(self, x, max_length = 512):
        
        desc = "PreProcessing dataset {}...".format('')
        
        if self.args['--lang']=='en':
            self.segment = 'joy sadness anger fear or suprise'
            self.labels = ['joy', 'sadness', 'anger', 'fear', 'suprise']
        elif self.args['--lang']=='ch':
            self.segment = '高兴 悲伤 愤怒 恐惧 厌恶 或 惊讶'
            self.labels = ['高兴', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶']
        elif self.args['--lang']=='de':
            self.segment = 'freude  traurigkeit  wut  ängstlichkeit  abstoßen  oder  überrascht'
            self.labels = ['freude', 'traurigkeit', 'wut', 'ängstlichkeit', 'abstoßen', 'überrascht']
        

        input_ids = [],input_length = [],lengths = []
        for i in tqdm(self.data, desc = desc):
            combined_text = f"{self.segment} [SEP] {i}"
        
            encoding = self.tokenizer(
                combined_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            input_id = encoding["input_ids"]
            input_ids.append(input_id)
            input_length.append(len(input_id))
            lengths.append(len(i))
            
            label_idxs = []
            for label in self.labels:
                tokens = self.tokenizer.convert_ids_to_tokens(input_id)
                idxs = [tokens.index(label) for label in self.labels if label in tokens]
                label_idxs.append(idxs)
            
            
        
        return input_ids, input_length, lengths, label_idxs