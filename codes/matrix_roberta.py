import torch, json, evaluate, random
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns

# Load Model & Tokenizer
model_path = './results/model-2-roberta-large'
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Load Test Data Set
test_file_path = './data/twentyquestions-test-preprocess-0.jsonl'
test_file_1_path = './data/twentyquestions-test-preprocess-1.jsonl'
test_file_2_path = './data/twentyquestions-test-preprocess-2.jsonl'
new_dataset = load_dataset("json", data_files={"test": test_file_2_path})

# Tokenization & Preprocess
def tokenization(dataset):
    return tokenizer(dataset['subject'], dataset['question'], padding='max_length', truncation=True)

def postprocess(dataset):
    columns_to_remove = ['subject', 'question', 'answer', 'quality_labels', 'score', 'high_quality', 'labels', 'is_bad', 'true_votes', 'majority', 'subject_split_index', 'question_split_index']
    new_dataset = dataset.remove_columns(columns_to_remove)
    new_dataset = new_dataset.rename_column('label', 'labels')
    new_dataset.set_format("torch")
    return new_dataset

tokenized_new = new_dataset.map(tokenization, batched=True)
tokenized_new = postprocess(tokenized_new)

# Set Device to CUDA
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Test Model
test_dataloader = DataLoader(tokenized_new['test'])
test_labels = []
test_preds = []

for batch in tqdm(test_dataloader):
    batch = {k: v.to(device) for k,v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    batch['preds'] = preds
    
    test_labels.append(batch['labels'].cpu())
    test_preds.append(batch['preds'].cpu())
    
# Make Confusion Matrix
def plot_confusion_matrix(confusion_matrix):
    """
    Plot a confusion matrix using Seaborn.
    
    Args:
        confusion_matrix (np.ndarray): The confusion matrix array.
        labels (list): List of class labels.
    """
    plt.figure(figsize=(8, 8))

    # Create the heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.savefig('matrix/matrix-2-roberta-large.png')
    
# Print Accuracy & F1
acc = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds, average='weighted')
print(f'Accuracy: {acc}, F1: {f1}')

# Plot Confusion Matrix
plot_confusion_matrix(confusion_matrix(test_labels, test_preds))