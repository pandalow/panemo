import pandas as pd
from datasets import Dataset
from utils import preprocess_function
from config import config


def load_data(args):
    train_df = pd.read_csv(f'data/{args.lang}/train.csv')
    dev_df = pd.read_csv(f'data/{args.lang}/dev.csv')
    test_df = pd.read_csv(f'data/{args.lang}/test.csv')

    return train_df, dev_df, test_df

def process(args, tokenizer):
    train_df, dev_df, test_df = load_data(args)

    all_data = {}

    for name, df in zip(["train", "dev", "test"], [train_df, dev_df, test_df]):
        texts = df["text"].tolist()
        if args.lang == 'en':
            labels = df[["anger", "fear", "joy", "sadness", "surprise"]].values
        else:
            labels = df[["anger", "disgust", "fear", "joy", "sadness", "surprise"]].values

        data = []
        for text, label in zip(texts, labels):
            if name == "test":
                data.append({"text": text})
            else:
                data.append({"text": text, "labels": label.tolist()})

        all_data[name] = data

    for name, data in all_data.items():
        dataset = Dataset.from_list(data)
        encoded_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names  # 确保原始数据被移除
        )
        all_data[name] = encoded_dataset

    return all_data["train"], all_data["dev"], all_data["test"]