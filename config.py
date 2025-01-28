
train_args = {
    "model_type": "seq2seq",
    'model_path': 'google/mt5-base',
    'file_path': '/content/drive/MyDrive/dataset/chn/train.csv',
    'lang': 'cn',
    'max_len': 512,
    'num_labels': 6,
    # 调大dropout，增强正则化
    'dropout': 0.2,
    'joint_loss': 'joint',
    # 减小alpha，让corr_loss占比更小
    'alpha': 0.05,
    'pool_type': 'mean',
    'batch_size': 4,
    'num_epochs': 10,
    'device': "cuda",
    'gradient_accumulation_steps': 4,
    'lr': 5e-6,
    'weight_decay': 0.01
}

eval_args = {
    'batch_size': 4,
    'device': "cuda"
}