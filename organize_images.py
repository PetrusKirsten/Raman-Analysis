
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def organize_images_by_pattern(folder_path, patterns):
    logging.info(f"Scanning folder: {folder_path}")
    
    for file_name in os.listdir(folder_path):

        full_file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(full_file_path):

            for pattern in patterns:

                if pattern in file_name:

                    target_folder = os.path.join(folder_path, pattern)

                    os.makedirs(target_folder, exist_ok=True)

                    new_file_path = os.path.join(target_folder, file_name)

                    shutil.move(full_file_path, new_file_path)

                    logging.info(f"Moved '{file_name}' to '{target_folder}'")

                    break

if __name__ == "__main__":

    # Define the folder containing the image files
    st = "./figures/maps/St CLs/bands_40to1785_w-bg_nearest"

    kc = "./figures/maps/St kC CLs/bands_40to1785_w-bg_nearest"

    ic = "./figures/maps/St iC CLs/bands_40to1785_w-bg_nearest"

    car = "./figures/maps/Carrageenans/bands_40to1785_w-bg_nearest"

    # Define the patterns to look for in file names
    name_patterns = ["480", "550", "805", "850", "941", "1220"]

    for folder in [car]:
        organize_images_by_pattern(folder, name_patterns)

    logging.info("Image organization completed successfully.")
