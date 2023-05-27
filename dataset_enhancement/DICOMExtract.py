import os
import glob
import shutil
import pydicom
from PIL import Image
import numpy as np
import time

def find_and_copy_dcm_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Initialize a counter
    count = 0
    
    # Recursively search for .dcm files in the source folder and its subdirectories
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.dcm'):
                # Get the full path of the DICOM file
                dcm_path = os.path.join(root, file)
                
                # Generate the new filename with count
                new_filename = f"CMMD_{count}"
                
                # Construct the destination path
                destination_path = os.path.join(destination_folder, new_filename + '.dcm')
                
                # Copy the DICOM file to the destination folder
                shutil.copy2(dcm_path, destination_path)
                
                # Increment the counter
                count += 1


def dicom_to_jpg(dicom_path, jpg_path):
    # Load DICOM image
    dicom = pydicom.dcmread(dicom_path)

    # Normalize pixel values
    pixel_array = dicom.pixel_array
    normalized_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    normalized_image = Image.fromarray(normalized_array.astype(np.uint8))

    # Save as JPEG image
    normalized_image.save(jpg_path, "JPEG", quality=100)


def resize_image(image_path, output_path):
    with Image.open(image_path) as image:
        resized_image = image.resize((512, 512))
        resized_image.save(output_path)


def resize_images_in_folder(folder_path):
    start_time = time.time()

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, filename)  # Same name for resized image

            resize_image(image_path, output_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Resizing completed in {:.2f} seconds.".format(elapsed_time))

# Set paths for the input DICOM folder and output JPEG folder
source_folder = '/Users/lun/Desktop/images/manifest-1684936432712/CBIS-DDSM'
dicom_folder = '/Users/lun/Desktop/images/DICOM'
jpeg_folder = '/Users/lun/Desktop/images/JPG'

# Find and copy DICOM files
find_and_copy_dcm_files(source_folder, dicom_folder)

# Iterate through DICOM files in the folder
for filename in os.listdir(dicom_folder):
    if filename.endswith(".dcm"):
        dicom_path = os.path.join(dicom_folder, filename)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(jpeg_folder, jpg_filename)

        # Convert DICOM to JPEG
        dicom_to_jpg(dicom_path, jpg_path)

# Resize the created JPEG images
resize_images_in_folder(jpeg_folder)