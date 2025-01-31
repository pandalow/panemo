import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


def custom_data_collator(batch):
    # 处理 input_ids, attention_mask
    batch_inputs = {key: torch.stack([torch.tensor(item[key]) for item in batch if key in item]) for key in batch[0] if key in ["input_ids", "attention_mask"]}

    # 只有在 batch 里存在 labels 时才处理 labels
    if "labels" in batch[0]:
        batch_inputs["labels"] = torch.stack([torch.tensor(item["labels"]) for item in batch])

    return batch_inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits))  # 使用 sigmoid 进行二分类
    predictions = (predictions > 0.3).int().cpu().numpy()  # 转换为 0/1 预测值

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")  # 计算宏平均 F1 分数
    print(classification_report(labels, predictions))  # 打印分类报告
    return {"accuracy": accuracy, "f1_macro": f1_macro}


def preprocess_function(examples, tokenizer):
    inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=256
    )

    if "labels" in examples and examples["labels"] is not None:
        inputs["labels"] = np.array(examples["labels"], dtype=np.float32).tolist()

    return inputs