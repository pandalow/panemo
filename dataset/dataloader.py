import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from utils import find_subsequence

class EmoDataset(Dataset):
    def __init__(self, args):
        """
        :param args: dict, 包含以下字段
          - file_path: 数据文件路径 (CSV)
          - lang: 'en', 'cn', or 'nl' (根据需要自行扩充)
          - model_path: HF模型名称/路径 (支持多语言)
          - max_len: int, tokenizer截断长度
        """
        super().__init__()
        self.args = args

        # 1) 构建分词器
        self.tokenizer = AutoTokenizer.from_pretrained(args['model_path'])

        self.max_len = args['max_len']

        # 2) 读取数据
        df = pd.read_csv(args['file_path'])
        self.texts = df['text'].tolist()

        # 3) 根据语言选择不同的情感标签与对应列
        if args['lang'] == 'en':
            # English
            self.label_matrix = df[['anger','fear','joy','sadness','surprise']].values
            self.label_names = ['anger','fear','joy','sadness','surprise']
        elif args['lang'] == 'cn':
            # Chinese
            self.label_matrix = df[['anger','disgust','fear','joy','sadness','surprise']].values
            self.label_names = ['愤怒','厌恶','害怕','高兴','悲伤','惊讶']
        else:
            # Dutch
            self.label_matrix = df[['anger','disgust','fear','joy','sadness','surprise']].values
            self.label_names = ['woede','walging','angst','vreugde','verdriet','verassing']

        self.tokenizer.add_special_tokens({'additional_special_tokens': 
                                   [f'[{lab}]' for lab in self.label_names]})
        
        # 4) 预处理，得到 input_ids, label_positions, labels
        self.input_ids, self.label_positions, self.labels = self.process_data()

    def process_data(self):
        # 分词标签
        label_subtokens = [
            self.tokenizer(label, add_special_tokens=False)["input_ids"] for label in self.label_names
        ]

        # 构造 prompt
        if self.args['lang'] == 'en':
            segment_a = "Detect the position of sentiment labels in the following text. Labels: " + " | ".join(self.label_names)
        elif self.args['lang'] == 'cn':
            segment_a = "检测情绪标签位置，标签为: " + " | ".join(self.label_names)
        else:
            segment_a = "Detecteer de locatie van emotielabels, labels: " + " | ".join(self.label_names)

        all_input_ids = []
        all_label_positions = []
        all_labels = []

        # 对每个文本进行处理
        for i, text in enumerate(self.texts):
            combined = f"{segment_a}\n{text}"
            enc = self.tokenizer(
                combined,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            ids_2d = enc["input_ids"]  # [1, seq_len]
            ids_1d = ids_2d[0].tolist()

            y = self.label_matrix[i]  # [num_labels]
            positions_for_sample = []

            # 为该条样本的每个标签找到位置
            for lab_sub in label_subtokens:
                start_pos = find_subsequence(ids_1d, lab_sub, self.tokenizer)
                if start_pos >= 0:
                    positions_for_sample.append((start_pos, len(lab_sub)))
                else:
                    positions_for_sample.append((-1, -1))  # 标记未找到

            all_input_ids.append(ids_1d)
            all_label_positions.append(positions_for_sample)
            all_labels.append(y)

        # 转为张量
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)      # [dataset_size, seq_len]
        all_label_positions = torch.tensor(all_label_positions, dtype=torch.long)  # [dataset_size, num_labels, 2]
        all_labels = torch.tensor(all_labels, dtype=torch.float)           # [dataset_size, num_labels]

        return all_input_ids, all_label_positions, all_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],          # [seq_len]
            self.labels[idx],             # [num_labels]
            self.label_positions[idx]     # [num_labels, 2]
        )