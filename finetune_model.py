import pandas as pd
from datasets import Dataset
from common import *
from transformers import (MarianMTModel,
                          MarianTokenizer,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments
                          )


# Function to load data from CSV
def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "src_lang": "source",
        "tgt_lang": "target"
        })
    return Dataset.from_pandas(df)


# Preprocess function for tokenizing the data
def preprocess_data(examples, tokenizer, max_length=128):
    inputs = examples['source']
    targets = examples['target']
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs,
                             max_length=max_length,
                             truncation=True,
                             padding="max_length"
                             )
    labels = tokenizer(targets,
                       max_length=max_length,
                       truncation=True,
                       padding="max_length"
                       )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# Load tokenizer and model from local directory
model_path = MODEL_PATH
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Load datasets
train_csv_path = TRAIN_CSV_PATH
val_csv_path = VAL_CSV_PATH

train_dataset = load_data_from_csv(train_csv_path)
val_dataset = load_data_from_csv(val_csv_path)

# Preprocess datasets
tokenized_train_dataset = train_dataset.map(
                                            lambda x: preprocess_data(
                                                x, tokenizer),
                                            batched=True
                                            )
tokenized_val_dataset = val_dataset.map(
                                        lambda x: preprocess_data(
                                            x, tokenizer),
                                        batched=True
                                        )

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_path,  # Save back to the same local directory
    evaluation_strategy="steps",  # Evaluate at specific steps
    eval_steps=500,  # Evaluation interval
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=1000,  # Save checkpoints every 1000 steps
    save_total_limit=3,  # Limit to 3 checkpoints
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    predict_with_generate=True,  # For text generation evaluation
    logging_dir="./logs",  # Log directory
    logging_steps=100,  # Logging interval
    load_best_model_at_end=True,  # Load the best model based on eval_loss
    metric_for_best_model="eval_loss",  # Metric to track
    greater_is_better=False,  # Lower eval_loss is better
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer
)

# Start training
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved successfully to: {model_path}")
