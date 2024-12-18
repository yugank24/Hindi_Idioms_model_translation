# from common import *
from transformers import MarianMTModel, MarianTokenizer

# Specify the path to your locally fine-tuned model
model_name = r"C:\\Users\\USIT\\HINDI_MODEL_FINAL\\Hindi_Idioms_model_translation\\fine_tuned_model\\opus-mt-hi-en"

# Load the pre-trained model and tokenizer from the local directory
# model_name = MODEL_PATH
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_text(text):
    # Tokenize the input text (Hindi to English)
    inputs = tokenizer.encode(text,
                              return_tensors="pt",
                              padding=True,
                              truncation=True)

    # Perform the translation
    translated = model.generate(inputs,
                                max_length=512,
                                num_beams=4,
                                early_stopping=True)

    # Decode the translated text (Hindi to English translation)
    translated_text = tokenizer.decode(translated[0],
                                       skip_special_tokens=True)
    return translated_text
