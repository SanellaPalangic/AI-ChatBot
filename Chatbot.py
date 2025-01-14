import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)

# Step 1: Prepare the Dataset
def load_dataset(file_path):
    """
    Load and prepare the dataset for training. The dataset should be in JSON format.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    dialogues = []
    for entry in data:
        dialogues.append(f"User: {entry['prompt']}\nBot: {entry['response']}")
    return Dataset.from_dict({"text": dialogues})

# Step 2: Load Pretrained Model and Tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize the embeddings


# Step 3: Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("algebra_data.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Define Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling; this is causal modeling
)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir="./algebra_chatbot",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=5e-5,
    warmup_steps=100
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Optional: Use validation dataset if available
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 7: Train the Model
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# Step 8: Save the Fine-Tuned Model
print("Saving the fine-tuned model...")
model.save_pretrained("./algebra_chatbot")
tokenizer.save_pretrained("./algebra_chatbot")
print("Model saved successfully!")

# Step 9: Test the Fine-Tuned Chatbot
print("\nTesting the chatbot interactively...\n")
chatbot = pipeline("text-generation", model="./algebra_chatbot", tokenizer="./algebra_chatbot")

# Interactive Chat Loop
print("Algebra Chatbot: Hello! I can help you with basic algebra. Type 'exit' to quit.\n")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Algebra Chatbot: Goodbye! Keep practicing algebra!")
        break
    response = chatbot(f"User: {user_input}\nBot:", max_length=100, num_return_sequences=1)
    bot_response = response[0]["generated_text"].split("Bot:")[-1].strip()
    print(f"Algebra Chatbot: {bot_response}")
