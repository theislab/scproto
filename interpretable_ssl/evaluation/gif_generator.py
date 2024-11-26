from PIL import Image, ImageDraw, ImageFont
import glob
import os

# Define the input folder
input_folder = "./temp-res/daii/"

# Patterns for non-finetuned and finetuned images
file_pattern_base = os.path.join(input_folder, "ref-e*-umap.png")
file_pattern_finetuned = os.path.join(input_folder, "ref-e*-umap_finetuned.png")

# Load images for each pattern in the specified order
image_files = sorted(glob.glob(file_pattern_base)) + sorted(glob.glob(file_pattern_finetuned))
images_with_text = []

# Loop through each image, add text, and store it in a list
for img_path in image_files:
    img = Image.open(img_path).convert("RGBA")
    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)

    # Choose the font size, color, and calculate position for bottom-right corner
    font_size = 200  # Much larger font
    text_color = "black"
    image_name = os.path.basename(img_path)

    # Try loading a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the position for bottom-right corner using textbbox
    text_bbox = draw.textbbox((0, 0), image_name, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = (img_with_text.width - text_width - 10, img_with_text.height - text_height - 10)

    # Write the image filename on the image
    draw.text(text_position, image_name, fill=text_color, font=font)

    # Append the modified image to the list
    images_with_text.append(img_with_text)

# Define the output folder and GIF filename
output_folder = "./temp-res/daii/"
output_gif = os.path.join(output_folder, "umap_finetuned_with_labels.gif")

# Save the images as a GIF
images_with_text[0].save(output_gif, save_all=True, append_images=images_with_text[1:], duration=500, loop=0)

print(f"GIF created and saved as {output_gif}")
