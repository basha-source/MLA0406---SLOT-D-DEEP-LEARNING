from PIL import Image
import numpy as np
import os

# Get image path from user
image_path = input("Enter image path: ").strip()

# Remove quotes if pasted
image_path = image_path.replace('"', '').replace("'", "")

# Check if file exists
if not os.path.exists(image_path):
    print("\nError: File not found!")
    print("You entered:", image_path)
    exit()

try:
    img = Image.open(image_path)

    pixels = np.array(img)

    print("\nImage loaded successfully!")
    print("Image Shape:", pixels.shape)

    print("\nPixel Values:")
    print(pixels)

except Exception as e:
    print("\nError:", e)