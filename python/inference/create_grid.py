import os
from PIL import Image, ImageDraw, ImageFont
import re

def calculate_grid_size(total_images):
    # Calculate the grid size (rows, cols) based on the total number of images
    cols = 5
    rows = (total_images + cols - 1) // cols
    return rows, cols

def numerical_sort(value):
    # Extract the last numerical part from the file path using regular expression
    numerical_part = re.findall(r'\d+', value.split('/')[-1])
    if numerical_part:
        return int(numerical_part[-1])
    return 0


def create_grid_image(image_folder, output_path, probabilities):
    images = []
    # Collect all image file paths in the folder and sort them alphabetically
    image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

    image_paths = sorted(image_paths, key=numerical_sort)

    # Load the images
    for file_path in image_paths:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(Image.open(file_path))
        
    print(images)
    
    # Calculate the grid size (rows, cols)
    total_images = len(images)
    rows, cols = calculate_grid_size(total_images)
    
    # Calculate the maximum width and height of the images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)
    
    # Create a new blank image with white background to hold the grid
    grid_width = max_width * cols
    grid_height = max_height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Create a draw object to add text to images
    draw = ImageDraw.Draw(grid_image)
    
    font_size = 80  # Adjust this value to change the font size
    font = ImageFont.truetype("/System/Library/Fonts/NewYork.ttf", font_size)

    
    for i, (img, prob) in enumerate(zip(images, probabilities)):
        row = i // cols
        col = i % cols
        grid_image.paste(img, (col * max_width, row * max_height))
    
        prob_text = f"Prob: {prob:.4f}"
        draw.text((col * max_width, (row + 1) * max_height - font_size - 10), prob_text, fill='black', font=font)
    
    grid_image.save(output_path)
    print(f"Grid image saved as {output_path}")

if __name__ == "__main__":
    image_folder = "/Users/raffaelepojer/Dev/RBN-GNN/models/triangle_10_8_6_20230725-152135/exp_41/graphs"
    
    output_path = "/Users/raffaelepojer/Dev/RBN-GNN/models/triangle_10_8_6_20230725-152135/exp_41/graphs"
    probabilities = [0.03053748793900013, 0.02477438747882843, 0.04797354340553284, 0.7326235175132751, 0.997011661529541, 0.9991599321365356, 0.9997318387031555, 0.9999178647994995]

    create_grid_image(image_folder, output_path, probabilities)
