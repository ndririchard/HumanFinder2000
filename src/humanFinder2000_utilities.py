# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:30:03 2023

@author: richardn
"""
import os
import cv2
import shutil
import numpy as np
from imgaug import augmenters as iaa


def load_images(directory):
    """
    Load images from a directory.

    Args:
        directory (str): Path to the directory containing thermal images.

    Returns:
        List of loaded thermal images.
    """
    images = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            image = cv2.imread(path)  
            images.append(image)
    return images

def normalize_images(images):
    """
    Normalize the pixel values of images to the range [0, 1].

    Args:
        images (list of numpy arrays): List of thermal images.

    Returns:
        List of normalized thermal images.
    """
    normalized_images = []
    for image in images:
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        normalized_images.append(normalized_image)
    return normalized_images

def resize_images(images, target_size):
    """
    Resize a list of images to a target size.

    Args:
        images (list of numpy arrays): List of thermal images.
        target_size (tuple): Target size as (width, height).

    Returns:
        List of resized thermal images.
    """
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, target_size)
        resized_images.append(resized_image)
    return resized_images

def augment_images(images, augmentation_factor):
    """
    Augment a list of images to increase data variety.

    Args:
        images (list of numpy arrays): List of thermal images.
        augmentation_factor (int): Number of augmented copies to create for each image.

    Returns:
        List of augmented thermal images.
    """
    augmenter = iaa.Sequential([
        iaa.Rotate((-10, 10)),  # Random rotation
        iaa.Fliplr(0.5),        # Horizontal flip with 50% probability
        iaa.GaussianBlur(sigma=(0, 3.0)),  # Random Gaussian blur
    ])
    
    augmented_images = []
    for image in images:
        augmented_images.extend(augmenter.augment_images([image] * augmentation_factor))
    
    return augmented_images

def copy_files_with_extension(source_directory, destination_directory, extension):
    """
    Copy files with a specific extension from a source 
    directory to a destination directory.

    Args:
        source_directory (str): Path to the source directory.
        destination_directory (str): Path to the destination directory.
        extension (str): File extension to filter for (e.g., 'rgb8').

    Returns:
        None
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Loop through files in the source directory
    for filename in os.listdir(source_directory):
        # Check if the file ends with the specified extension
        if filename.endswith(extension):
            source_file = os.path.join(source_directory, filename)
            destination_file = os.path.join(destination_directory, filename)

            try:
                # Copy the file to the destination directory
                shutil.copy(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")
            except Exception as e:
                print(f"Error copying {source_file}: {str(e)}")


