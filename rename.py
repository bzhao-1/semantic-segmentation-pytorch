import os
import shutil

# Define the base directory paths
data_dir = '/Users/benzhao/Downloads/data'
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')

# Loop through the 'training' and 'validation' subfolders
for subfolder in ['training']:
    # Get the list of image filenames in the 'images' folder
    image_folder = os.path.join(images_dir, subfolder)
    
    
    # Ensure the directories exist
    if not os.path.exists(image_folder) or not os.path.exists(annotation_folder):
        print(f"Warning: Folder {subfolder} does not exist!")
        continue
    
    # List of all images in the folder
    image_files = os.listdir(image_folder)

    
            
print("Renaming process completed.")