from transformers import BartTokenizer, BartForConditionalGeneration

# Step 1: Load the model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Step 2: Load your college notes (from a text file)
with open('college_notes.txt', 'r') as file:
    notes = file.read()

# Step 3: Tokenize the input text
inputs = tokenizer(notes, max_length=1024, return_tensors="pt", truncation=True)

# Step 4: Generate the summary
summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

# Step 5: Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Step 6: Save the summary to a text file
with open('summary.txt', 'w') as file:
    file.write(summary)

print("Summary saved to summary.txt")
