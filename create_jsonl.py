import json
import os
import re

# Set the path to your image folder
image_folder = 'data/kanji_images'

# The path to the .jsonl file you want to create
jsonl_file_path = 'data/kanji_images/metadata.jsonl'

# List all files in the image folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and re.match(r'.*\.(jpg|jpeg|png|gif|bmp|tiff|tif|svg|webp)', f)]

# Open the .jsonl file in write mode
with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
    for image_file in image_files:
        # Create a caption based on the filename (you can modify this logic as needed)
        caption = os.path.splitext(image_file)[0].split('<i>')[0].replace('<w>', ', ')

        # Create a JSON object for this image
        metadata = {
            "file_name": image_file,
            "text": caption
        }

        # Write the JSON object to the .jsonl file
        jsonl_file.write(json.dumps(metadata) + '\n')