import numpy as np  
import os  
import shutil  # Import the shutil library for file operations

# Function to load images into a NumPy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Function to calculate the area of the displayed logo
def calculate_area(x1, y1, x2, y2):
    xDiff = abs(x1 - x2)
    yDiff = abs(y1 - y2)
    area = xDiff * yDiff
    return area

# Function to calculate the shortest and largest area of displayed logos
def shortest_longest_area(area_list):
    area_list.sort()
    shortest = area_list[0]
    longest = area_list[-1]
    response = {
        "shortest": shortest,
        "longest": longest
    }
    return response

# Function to delete and create a folder
def delete_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Remove the folder if it exists
        os.makedirs(folder_path, 0o755)  # Create the folder with appropriate permissions
    else:
        os.makedirs(folder_path, 0o755)  # Create the folder if it doesn't exist
