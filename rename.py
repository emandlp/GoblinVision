import os

def add_suffix_to_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate through each file
    for filename in files:
        # Split the filename into base name and extension
        base_name, extension = os.path.splitext(filename)

        # Construct the new filename with 'a' added at the end
        new_filename = os.path.join(directory, base_name + 'a' + extension)

        # Rename the file
        os.rename(os.path.join(directory, filename), new_filename)
        print(f"Renamed {filename} to {base_name + 'a' + extension}")


# Example usage:
directory_path = r"C:\Users\emand\PycharmProjects\yolo5\runs\detect\exp12\labels"
add_suffix_to_files(directory_path)