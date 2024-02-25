import xml.etree.ElementTree as ET
from cairosvg import svg2png
from PIL import Image
import io
import re
from tqdm import tqdm
import os

# File paths for the provided XML files
kanjidic2_file_path = 'data/kanjidic2.xml'
kanjivg_file_path = 'data/kanjivg-20220427.xml'

# Parse KANJIDIC2 XML
kanjidic2_tree = ET.parse(kanjidic2_file_path)
kanjidic2_root = kanjidic2_tree.getroot()

# Extracting a few Kanji characters and their information from KANJIDIC2
kanji_info = {}
for character in tqdm(kanjidic2_root.findall('character'),desc='Extracting Kanji Information'):
    entry = {
        'kanji': character.find('literal').text,
        'meanings': [mean.text for mean in character.findall('reading_meaning/rmgroup/meaning') if mean.attrib.get('m_lang') is None],
    }
    kanji_info[entry['kanji']] = entry

# Parse KanjiVG XML
kanjivg_tree = ET.parse(kanjivg_file_path)
kanjivg_root = kanjivg_tree.getroot()

# Mapping Kanji to SVG paths from KanjiVG
kanji_svg_paths = {}
for kanji in tqdm(kanji_info.values(), desc='Mapping Kanji to SVG Paths'):
    for character in kanjivg_root:
        # The id attribute in KanjiVG files follows the format 'kvg:kanji_codepoint'
        if character.attrib['id'].endswith(hex(ord(kanji['kanji']))[2:].upper()):
            kanji_svg_paths[kanji['kanji']] = ET.tostring(character, encoding='unicode')
            break

# Transfer the SVG data to a PNG image
kanji_head = r'<(/?)kanji[^>]*>'
svg_head = '<svg xmlns:kvg="http://kanjivg.tagaini.net" width="109" height="109" style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'
svg_tail = '</svg>'
for kanji, svg_data in tqdm(kanji_svg_paths.items(), desc='Converting SVG to PNG'):
    svg_data = re.sub(kanji_head, '', svg_data)
    svg_data = svg_data.replace('ns0', 'kvg')
    svg_data = svg_head + svg_data + svg_tail
    png_data = svg2png(bytestring=svg_data.encode('utf-8'), output_width=500, output_height=500)
    all_meaning = []
    for meaning in tqdm(kanji_info[kanji]['meanings'], leave=False):
        meaning = re.sub(r'[^\w\s]', '', meaning)
        img_name = meaning
        all_meaning.append(meaning)
        # Convert PNG data to an image for viewing
        os.makedirs('data/kanji_images', exist_ok=True)
        try:
            image = Image.open(io.BytesIO(png_data))
            # image.show()
            image.save(f'data/kanji_images/{img_name}.png')  # Save the image to a file
        except Exception as e:
            print(f'Error saving {kanji_info[kanji]['kanji']}: {img_name}.png to PNG: {e}')
            continue
    img_name = "<w>".join(all_meaning)
    os.makedirs('data/kanji_images', exist_ok=True)
    try:
        image = Image.open(io.BytesIO(png_data))
        # image.show()
        image.save(f'data/kanji_images/{img_name}.png')  # Save the image to a file
    except Exception as e:
        print(f'Error saving {kanji_info[kanji]['kanji']}: {img_name}.png to PNG: {e}')
        continue