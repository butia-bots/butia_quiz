from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name and the directory to save it
model_name = "google/gemma-2b"
save_directory = "resources/models/gemma_model"

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_directory)

print("Model and tokenizer downloaded and saved locally.")