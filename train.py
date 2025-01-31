import torch
from models.model import CustomModel
from transformers import Trainer, TrainingArguments,AutoTokenizer,AutoModel
from utils import custom_data_collator, compute_metrics
from config import config
from dataset.dataloader import process

model_name = 'hfl/llama-3-chinese-8b-instruct-v3'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def get_training_args(output_dir="./results", quantization_bits=8):
    gradient_checkpointing = True  # 启用梯度检查点节省显存

    # 选择优化器
    optim = "adamw_8bit" if quantization_bits in [8, 4] else "adamw_torch"

    # 是否启用 fp16 训练（8-bit 或 4-bit 量化时禁用）
    fp16 = False
    bf16 = False

    # 4-bit 量化需要额外支持
    load_in_4bit = quantization_bits == 4
    bnb_4bit_compute_dtype = torch.float16 if load_in_4bit else None
    bnb_4bit_quant_type = "nf4" if load_in_4bit else None

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_dir=config.logging_dir,
        report_to=config.report_to,
        dataloader_num_workers=config.dataloader_num_workers,
        run_name=config.run_name,
    )
    return training_args

quantization_bits = config.quantization_bits
training_args = get_training_args(quantization_bits=quantization_bits)

model = CustomModel(model_name=model_name,
                    num_labels=6)

train_dataset,dev_dataset,test_dataset = process(config,tokenizer)


# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,      # 训练数据集
    eval_dataset=dev_dataset,   # 验证数据集
    data_collator=custom_data_collator, # 数据收集函数
    tokenizer=tokenizer,                # tokenizer
    compute_metrics=compute_metrics,    # 计算评估指标
)

trainer.train()

