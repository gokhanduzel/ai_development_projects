from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Keeping track of conversation history
conversation_history = []

# Encoding the conversation history
history_string = "\n".join(conversation_history)

# Fetch prompt from user
input_text ="hello, how are you doing?"

# Tokenization of user prompt and chat history
inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
print(inputs)

tokenizer.pretrained_vocab_files_map

# Generate output from the model
outputs = model.generate(**inputs)
print(outputs)

# Decode output
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(response)