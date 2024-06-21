import torch, json, os
import numpy as np
from copy import deepcopy
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, f1_score

# WANDB settings
os.environ["WANDB_PROJECT"] = "twentyquestions-bert"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load & Preprocess data
train_file_path = './data/twentyquestions-train-preprocess-2.jsonl'
dev_file_path = './data/twentyquestions-dev-preprocess-2.jsonl'
test_file_path = './data/twentyquestions-test-preprocess-2.jsonl'

new_dataset = load_dataset("json", data_files={"train": train_file_path, "eval": dev_file_path, "test": test_file_path})

# Load Tokenizer & Mapping
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenization(dataset):
    return tokenizer(dataset['subject'], dataset['question'], padding='max_length', truncation=True)

tokenized_new = new_dataset.map(tokenization, batched=True)

# Remove & Rename Features
def postprocess(dataset):
    columns_to_remove = ['subject', 'question', 'answer', 'quality_labels', 'score', 'high_quality', 'labels', 'is_bad', 'true_votes', 'majority', 'subject_split_index', 'question_split_index']
    new_dataset = dataset.remove_columns(columns_to_remove)
    return new_dataset

tokenized_new = postprocess(tokenized_new)

# Initialize Model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Generate Metric
def compute_metrics(predictions):
    logits, labels = predictions
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
     # Consider Lable Imbalance
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# For Train Accuracy and F1
class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

# Set TrainingArguments
training_args = TrainingArguments(
    report_to='wandb',
    run_name='twentyquestions-model2-roberta-base-batch32-4e-5-epoch4',
    output_dir='./results/checkpoints',         
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=2,
    num_train_epochs=4,           
    learning_rate=4e-5,
    # weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    do_eval=True,
    load_best_model_at_end=True,
    warmup_steps=100,               
    logging_steps=20        
)

# Make Trainer Object
trainer = Trainer(
    model=model,                     
    args=training_args,              
    train_dataset=tokenized_new['train'],     
    eval_dataset=tokenized_new['eval'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics 
)

# Train a Model
trainer.add_callback(CustomCallback(trainer))
train_result = trainer.train()
trainer.save_model('./results/model-2-roberta-base')