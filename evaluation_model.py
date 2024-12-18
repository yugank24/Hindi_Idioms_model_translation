import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import evaluate
from common import *
import os 

# Load BLEU Metric
bleu = evaluate.load(SACREBLEU)


# Function to Compute BLEU Score
def compute_bleu(predictions, references):
    # SacreBLEU expects references as a list of lists
    results = bleu.compute(predictions=predictions,
                           references=[[ref] for ref in references])
    return results["score"]


# Load Model and Tokenizer
model_path = MODEL_PATH
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Load Validation Dataset
dataset_folder = DATASET_FOLDER
val_csv_filename = VAL_CSV_FILENAME
val_csv_path = os.path.join(dataset_folder, val_csv_filename)
val_data = pd.read_csv(val_csv_path)

# Ensure the CSV has "src_lang" and "tgt_lang" columns
if SRC_LANG not in val_data.columns or TGT_LANG not in val_data.columns:
    raise ValueError(f"Validation CSV must contain {SRC_LANG} and {TGT_LANG} columns.")


# Function to Translate Text
def translate_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Generate output
    outputs = model.generate(**inputs, max_length=128)
    # Decode and return the translated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Generate Predictions
print("Generating translations for the validation dataset...")
predictions = [translate_text(src) for src in val_data["src_lang"]]
references = val_data[TGT_LANG].tolist()

# Compute BLEU Score
print("Computing BLEU score...")
bleu_score = compute_bleu(predictions, references)
print(f"BLEU Score on Validation Set: {bleu_score:.2f}")

# Save Results
results_df = pd.DataFrame({
    "Source": val_data[SRC_LANG],
    "Reference": references,
    "Prediction": predictions
})

results_df.to_csv(TRANSLATION_RESULTS, index=False)
print(F"Translation results saved to {TRANSLATION_RESULTS}.")
