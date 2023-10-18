import os
from PIL import Image

def search_image_by_name(directory, image_name):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == image_name.lower():
                return True

    return False

def remove_image_by_name(directory, image_name):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower() == image_name.lower():
                image_path = os.path.join(root, file)
                os.remove(image_path)
                return True

    return False

def get_image_names(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"]  # Add more extensions as needed
    image_names = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_names.append(file)

    return image_names

if __name__ == "__main__":
    directory_path_th0 = r"data\images_th0"  # Replace with the actual directory path
    image_names_th0 = get_image_names(directory_path_th0)
    directory_path_th1 = r"data\images_th1"  # Replace with the actual directory path
    image_names_th1 = get_image_names(directory_path_th1)
    directory_path_depth = r"data\images_depth"  # Replace with the actual directory path
    image_names_depth = get_image_names(directory_path_th0)
    directory_path_rgb = r"data\images_rgb"
    image_names_rgb = get_image_names(directory_path_rgb)
    for name in image_names_th1:
        rgb_name = name[0 : 7] + 'rgb8' + name[15 :]
        print(rgb_name)
        print(search_image_by_name(directory_path_rgb, rgb_name))
        if not search_image_by_name(directory_path_rgb, rgb_name):
            th0_name = name[0 : 7] + 'thermal0' + name[15 :]
            depth_name = name[0 : 7] + 'depth16' + name[15 :]
            remove_image_by_name(directory_path_th0, th0_name)
            remove_image_by_name(directory_path_th1, name)
            remove_image_by_name(directory_path_depth, depth_name)
            print(th0_name, depth_name, name)

