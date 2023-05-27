from PIL import Image
import time
import os
import cv2

# Path to the folder containing the images
folder_path = "/Users/lun/Desktop/images/JPG"

# Function to apply CLAHE to an image
def apply_clahe(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale if it's in color
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create CLAHE object with desired parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to the image
    enhanced_image = clahe.apply(image)

    return enhanced_image

# Start the timer
start_time = time.time()

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith('.jpg'):
        # Open the image file
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        
        # Convert the image to RGB mode (remove alpha channel)
        image = image.convert("RGB")
        name, extension = os.path.splitext(filename)
        
        # Rotate the image 11 times and save each rotated copy
        for i in range(11):
            # Rotate the image by 30 degrees
            rotated_image = image.rotate(30 * i + 30)
            
            # Generate a new filename for the rotated image
            rotated_filename = f"{name}_{30 * i + 30}.jpg"
            rotated_image_path = os.path.join(folder_path, rotated_filename)

            # Save the rotated image
            rotated_image.save(rotated_image_path)
            
            # Apply CLAHE to the image
            contrast_image = apply_clahe(rotated_image_path)
        
            # Construct the output path for the contrast-enhanced image
            contrast_filename = f"{name}_{30 * i + 30}_contrast.jpg"
            contrast_image_path = os.path.join(folder_path, contrast_filename)
        
            # Save the contrast-enhanced image
            cv2.imwrite(contrast_image_path, contrast_image)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

print(f"Image rotation, saving and contrast enhancement applied to all images in {elapsed_time:.2f} seconds.")