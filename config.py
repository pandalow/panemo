# config.py
import argparse
import torch

config = argparse.Namespace(
    model_name='hfl/llama-3-chinese-8b-instruct-v3',
    lang='chn',
    output_dir="./results",
    quantization_bits=8,
    gradient_checkpointing=True,
    optim="adamw_8bit" if 8 in [8, 4] else "adamw_torch",
    fp16=False,
    bf16=False,
    load_in_4bit=8 == 4,
    bnb_4bit_compute_dtype=torch.float16 if 8 == 4 else None,
    bnb_4bit_quant_type="nf4" if 8 == 4 else None,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_dir="./logs",
    report_to="none",
    dataloader_num_workers=4,
    run_name="llama8b-sentiment",
    quantization_bits=8,
    num_labels=6,
    load_in_8bit=True,

)
