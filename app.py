import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Load fine-tuned model and tokenizer
model = GPTNeoForCausalLM.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Set model to evaluation mode
model.eval()

st.title("Financial Chatbot")

user_input = st.text_input("Enter your query:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=10, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
