import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments


df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# Step 2: Encode Labels
label_encoder = LabelEncoder()
df["intent_encoded"] = label_encoder.fit_transform(df["intent"])

# Step 3: Tokenize Inputs
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(list(df["instruction"]), truncation=True, padding=True, max_length=128)
encodings["labels"] = list(df["intent_encoded"])  # Add encoded labels

# Step 4: Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key != "labels"}
        item["labels"] = torch.tensor(int(self.encodings["labels"][idx]))
        return item

# Step 5: Split Data into Train and Validation (Option for Evaluation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the training set
train_encodings = tokenizer(list(train_df["instruction"]), truncation=True, padding=True, max_length=128)
train_encodings["labels"] = list(train_df["intent_encoded"])  # Add encoded labels
train_dataset = CustomDataset(train_encodings)

# Tokenize the validation set
val_encodings = tokenizer(list(val_df["instruction"]), truncation=True, padding=True, max_length=128)
val_encodings["labels"] = list(val_df["intent_encoded"])  # Add encoded labels
val_dataset = CustomDataset(val_encodings)

# Step 6: Load Model
num_labels = len(label_encoder.classes_)
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
import os
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B integration

# Step 7: Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Option for evaluation (use "no" for no evaluation)
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Step 8: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Pass validation dataset here for evaluation
    tokenizer=tokenizer,
)

# Step 9: Train the Model
trainer.train()

# Step 10: Save the Model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Step 11: Test the Model (Optional)
test_text = "	i want help cancelling purchase A123456"
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Intent: {label_encoder.inverse_transform([predicted_class])[0]}")

