# train_extractor.py
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import AutoTokenizer
def main():
    # --- 1. 加载和预处理数据 ---

    # 模型检查点，mengzi-t5-base 是一个强大的中文T5模型
    model_checkpoint = "Langboat/mengzi-t5-base"
    
    # 加载分词器
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


    tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base", use_fast=False)
    # 加载数据集（假设你的数据都在一个文件里，实际应用中可以拆分）
    # 如果你有 train.jsonl 和 validation.jsonl，可以这样加载：
    # raw_datasets = load_dataset('json', data_files={'train': 'train.jsonl', 'validation': 'validation.jsonl'})
    raw_datasets = load_dataset('json', data_files='triplets_dataset.jsonl').shuffle(seed=42)
    # 临时拆分出训练集和验证集
    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)

    # 为模型的输入添加任务前缀，这能让模型更好地理解任务
    prefix = "抽取三元组: "

    # 定义预处理函数
    def preprocess_function(examples):
        # 准备模型的输入（原文）
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # 准备模型的标签（目标JSON字符串）
        labels = tokenizer(text_target=examples["output"], max_length=256, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 对整个数据集进行预处理
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    print("数据集预处理完成！")
    print(f"训练集样本数: {len(tokenized_datasets['train'])}")
    print(f"验证集样本数: {len(tokenized_datasets['test'])}")

    # --- 2. 配置训练 ---

    # 加载预训练模型
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # 定义数据整理器，用于在训练时动态填充批次数据
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 定义训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="./triplet_extractor_model",   # 模型输出和保存的目录
        num_train_epochs=10,                      # 训练轮次
        per_device_train_batch_size=8,            # 训练批次大小
        per_device_eval_batch_size=8,             # 评估批次大小
        evaluation_strategy="epoch",              # 每个 epoch 结束后进行评估
        save_strategy="epoch",                    # 每个 epoch 结束后保存模型
        logging_dir='./logs',                     # 日志目录
        logging_steps=10,                         # 每10步记录一次日志
        predict_with_generate=True,               # 在评估时使用generate生成文本
        learning_rate=3e-5,                       # 学习率
        weight_decay=0.01,                        # 权重衰减
        load_best_model_at_end=True,              # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",        # 以验证集损失作为最佳模型的标准
        greater_is_better=False,                  # 损失越小越好
        fp16=torch.cuda.is_available(),           # 如果有CUDA，使用半精度训练加速
    )

    # 创建 Trainer 实例
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 3. 执行训练 ---
    print("开始训练模型...")
    trainer.train()
    print("训练完成！")

    # --- 4. 保存最终模型 ---
    final_model_path = "./final_triplet_extractor"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")


if __name__ == "__main__":
    main()