import os
import csv

# Define paths
txt_folder = "E:\image labelling\labelled images"  #.txt files
image_folder = "E:\image labelling\images_dataset_mlsc"  #.jpg images
output_csv = "dataset.csv"

# Collect all txt file paths
txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

# Open CSV file for writing
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(["Label", "Feature1", "Feature2", "Feature3", "Feature4"])

    # Process each text file
    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)

        # Find corresponding image file (.jpg)
        image_file = os.path.splitext(txt_file)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_file)

        # Skip and delete corresponding image if txt file is empty
        if os.path.getsize(txt_path) == 0:
            print(f"Skipping empty file and deleting image: {txt_file} -> {image_file}")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            else:
                print(f"Image not found: {image_file}, skipping deletion.")
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

            # Ensure we have exactly 5 values (1 label + 4 features)
            if len(values) != 5:
                print(f"Skipping malformed file: {txt_file} -> {line}")
                continue

            try:
                label = int(values[0])  # Convert first value to integer (label)
                features = list(map(float, values[1:]))  # Convert rest to floats

                # Write to CSV
                csv_writer.writerow([label] + features)
            except ValueError as e:
                print(f"Skipping file with invalid data: {txt_file} -> {e}")

print(f"CSV file '{output_csv}' created successfully!")
