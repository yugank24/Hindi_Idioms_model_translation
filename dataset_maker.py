import os
import csv
from common import *

# Define the number of target rows and other settings
target_rows = 1500
idiom_list = list(IDIOM_DICTIONARY.items())
repeated_data = []

# Repeat the data until the required number of rows is reached
while len(repeated_data) < target_rows:
    repeated_data.extend(idiom_list)
repeated_data = repeated_data[:target_rows]

# Define the folder and file name
dataset_folder = DATASET_FOLDER
csv_filename = CSV_FILENAME
file_path = os.path.join(dataset_folder, csv_filename)

# Ensure the folder exists
os.makedirs(dataset_folder, exist_ok=True)

# Write the data to the CSV file in the specified folder
with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([SRC_LANG, TGT_LANG])
    for hindi_idiom, english_translation in repeated_data:
        writer.writerow([hindi_idiom, english_translation])

print(f"CSV file '{file_path}' has been created.")
