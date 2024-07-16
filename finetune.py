import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, AdamW, get_scheduler, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
dataset = load_dataset('financial_phrasebank', 'sentences_allagree', split='train')

# Inspect dataset to find the correct key
print(dataset.column_names)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['sentence', 'label'])

# Prepare DataLoader with padding
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_loader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)

# Set up LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning function
def fine_tune_model(train_loader, model, tokenizer, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # Move tensors to the appropriate device
            inputs = {key: value.to(device) for key, value in batch.items() if key in tokenizer.model_input_names}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            if loss is None:
                continue

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

# Run fine-tuning
fine_tune_model(train_loader, model, tokenizer)

# Save the fine-tuned model
model.save_pretrained("fine-tuned-gpt-neo")
tokenizer.save_pretrained("fine-tuned-gpt-neo")

# Streamlit UI for Q&A (if needed)
import streamlit as st

st.title("Financial Chatbot")

user_input = st.text_input("You: ", "Type your question here...")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Chatbot: {response}")
