import os
import csv
import shutil

txt_folder = "E:/image labelling/labelled images"  # Folder containing .txt files
image_folder = "E:/image labelling/images_dataset_mlsc"  # Folder containing original images
output_folder = "E:/image labelling/annotated_images"  # Folder to store labelled images
output_csv = "dataset_new.csv"

os.makedirs(output_folder, exist_ok=True)

# Collect all txt file paths
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

# Open CSV file for writing
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    # Writing header
    csv_writer.writerow(["Image_Name", "Label"])

    # Process each text file
    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)

        # Find corresponding image file (.jpg)
        image_file = os.path.splitext(txt_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # Skip and delete corresponding image if txt file is empty
        if os.path.getsize(txt_path) == 0:
            print(f"Skipping empty file and deleting image: {txt_file} -> {image_file}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            continue  # Skip processing this .txt file

        # Read the content of the text file
        with open(txt_path, "r") as file:
            line = file.readline().strip()

            # Ensure line is not empty
            if not line:
                print(f"Skipping file with no data and deleting image: {txt_file} -> {image_file}")
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Deleted: {image_path}")
                continue  # Skip processing this .txt file

            # Split by whitespace (space, tab, etc.)
            values = line.split()

            # Ensure at least 1 value (label) is present
            if len(values) < 1:
                print(f"Skipping malformed file: {txt_file} -> {line}")
                continue

            try:
                label = int(values[0])  # Convert first value to integer (label)

                # Copy the image to the annotated dataset folder
                if os.path.exists(image_path):
                    shutil.copy(image_path, output_image_path)
                    print(f"Copied {image_file} to {output_folder}")

                    # Write to CSV
                    csv_writer.writerow([image_file, label])
                else:
                    print(f"Image not found: {image_file}, skipping.")

            except ValueError as e:
                print(f"Skipping file with invalid data: {txt_file} -> {e}")

print(f"CSV file '{output_csv}' created successfully with image names and labels!")
print(f"Annotated images are saved in '{output_folder}'.")
