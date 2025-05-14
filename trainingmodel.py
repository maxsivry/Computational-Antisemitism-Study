from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from huggingface_hub import notebook_login
from sklearn.model_selection import train_test_split

notebook_login()
# If the dataset is gated/private, make sure you have run huggingface-cli login
# dataset = load_dataset("mivry/annotated_training_antisem")

original_df = pd.read_csv("training1.csv")

# Split the DataFrame into two halves
train_df, test_df = train_test_split(original_df, test_size=0.5, random_state=42)
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# df = pd.read_csv("training.csv")
# # Create a dictionary for the dataset
# train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# # Create a dictionary for the dataset
# data_dict = {
#     'train': [{'label': int(label), 'text': str(pure_text)} for label, pure_text in zip(train_df['label'], train_df['pure_text'])],
#     'test': [{'label': int(label), 'text': str(pure_text)} for label, pure_text in zip(test_df['label'], test_df['pure_text'])],
# }
# # Create a datasets.Dataset object
# dataset = Dataset.from_csv('training1.csv', split=['train', 'test'])
# print(dataset.splits)

# dataset = Dataset.from_dict("training1.csv")
# # Print the first few examples of the dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
print(dataset['train'][0])
print(dataset['test'][0])


# # Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# # Preprocess function to tokenize text and truncate sequences
def preprocess_function(examples):
    return tokenizer(examples["pure_text"], truncation=True)

# # Tokenize your dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# # Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# # Evaluation metric
accuracy = evaluate.load("accuracy")
print("accuracy evaluation metric")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# # Define id2label and label2id
id2label = {0: "NA", 1: "A"}  
label2id = {"NA": 0, "A": 1}  


# # Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="antisemitism_model",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
trainer.push_to_hub()