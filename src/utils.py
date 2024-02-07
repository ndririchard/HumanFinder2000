import re
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import SAM, YOLO
from sympy import symbols, Eq, solve
from sklearn.linear_model import LinearRegression

# GLOBALS VARIABLES
th_dirs = [
    "D:\IDU5\HumanFinder2000\data\images_th0",
    "D:\IDU5\HumanFinder2000\data\images_th1",
]

rgb = [
    [569, 199],
    [645, 126],
    [302, 205]
]

th0 = [
    [447, 170],
    [544, 110],
    [94, 188]
]

th1 = [
    [663, 147],
    [754, 85],
    [293, 155]
]

"""rgb_th0 = np.array([
    [0.17001, -0.050397],
    [-0.016913, 0.24084]
    ])

rgb_th1 = np.array([
    [0.22486 ,  -0.0191],
    [-0.013638 ,    0.22081]
    ])"""

rgb_th0 = np.array(
    [
        [1.0446527594018822, -0.9504409974635526],
        [-0.010104249387085575, 0.9132869544971859],
    ]
)

rgb_th1 = np.array(
    [
        [0.11605030915131691, 0.3961926598432364],
        [-0.017737377350909356, 0.32507971119315626],
    ]
)
SIZE = (848, 480)


# GLOBAL FUNCTION
def solver(A, B):
    a, b, c, d = symbols('a b c d')
    eqs = []
    eq1 = Eq(569*a + 199*b, 447)
    eq2 = Eq(569*c + 199*d, 170)
    eq3 = Eq(645*a + 126*b, 544)
    eq4 = Eq(645*c + 126*d, 110)
    eq5 = Eq(302*a + 205*b, 94)
    eq6 = Eq(302*c + 205*d, 188)

    # Résoudre le système d'équations
    solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6), (a, b, c, d))
    #filtered_solutions = [sol for sol in solutions if sol[a] * sol[d] - sol[c] * sol[b] != 0]

    print(solutions)

def resize_image(image, size):
    return cv2.resize(image, size)

def annotate_images_with_resized_boxes(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    image_files = [
        f
        for f in os.listdir(input_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    for image_file in image_files:
        input_image_path = os.path.join(input_directory, image_file)
        original_image = cv2.imread(input_image_path)
        image = resize_image(original_image, SIZE)
        image_with_box = image.copy()
        cv2.namedWindow("Annotation d'image")

        box_start = None
        drawing_box = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal box_start, drawing_box

            if event == cv2.EVENT_LBUTTONDOWN:
                box_start = (x, y)
                drawing_box = True

            elif event == cv2.EVENT_LBUTTONUP:
                cv2.rectangle(image_with_box, box_start, (x, y), (0, 255, 0), 2)
                drawing_box = False

                cv2.imshow("Annotation d'image", image_with_box)

        cv2.setMouseCallback("Annotation d'image", mouse_callback)

        while True:
            cv2.imshow("Annotation d'image", image_with_box)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        output_annotated_path = os.path.join(output_directory, image_file)
        cv2.imwrite(output_annotated_path, image_with_box)
        cv2.destroyAllWindows()

def dot(M, v):
    res = []
    for _ in M:
        a = _[0] * v[0] + _[1] * v[1]
        res.append(a)
    return res

def convert_xy_to_th(A, B):
    x1, y1, x2, y2 = [], [], [], []
    for index in range(len(A)):
        x1.append(A[index][0] / A[index][1])
        y1.append(B[index][0] / A[index][1])
        y2.append(B[index][1] / A[index][1])

    x1 = np.array(x1).reshape((-1, 1))

    l1 = LinearRegression()
    l2 = LinearRegression()

    l1.fit(x1, y1)
    l2.fit(x1, y2)

    # Printing model 1 coefficients
    # M1 = np.array([l1.coef_[0], l1.intercept_], [l2.coef_[0], l2.intercept_])
    print(f"{l1.coef_[0]} {l1.intercept_}")
    print(f"{l2.coef_[0]} {l2.intercept_}")

def search_image_by_name(directory, image_name):
    """
    Search for an image file by name in the specified directory and its subdirectories.

    Parameters:
    - directory (str): The root directory to start the search.
    - image_name (str): The name of the image file to search for.

    Returns:
    - bool: True if the image file is found, False otherwise.
    """
    # Iterate through the directory and its subdirectories using os.walk
    for root, _, files in os.walk(directory):
        # Iterate through the files in the current directory
        for file in files:
            # Compare file names, case-insensitive
            if file.lower() == image_name.lower():
                # Return True if the image file is found
                return True

    # Return False if the image file is not found in the specified directory or its subdirectories
    return False

def remove_image_by_name(directory, image_name):
    """
    Remove an image file by name from the specified directory and its subdirectories.

    Parameters:
    - directory (str): The root directory to start the search.
    - image_name (str): The name of the image file to remove.

    Returns:
    - bool: True if the image file is found and removed, False otherwise.
    """
    # Iterate through the directory and its subdirectories using os.walk
    for root, _, files in os.walk(directory):
        # Iterate through the files in the current directory
        for file in files:
            # Compare file names, case-insensitive
            if file.lower() == image_name.lower():
                # Construct the full path to the image file
                image_path = os.path.join(root, file)
                # Remove the image file
                os.remove(image_path)
                # Return True if the image file is found and removed
                return True

    # Return False if the image file is not found in the specified directory or its subdirectories
    return False


def get_image_names(directory):
    """
    Get a list of image file names in the specified directory and its subdirectories.

    Parameters:
    - directory (str): The root directory to start the search.

    Returns:
    - list: A list of image file names.
    """
    # Define supported image file extensions
    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    ]  # Add more extensions as needed
    # Initialize an empty list to store image file names
    image_names = []

    # Iterate through the directory and its subdirectories using os.walk
    for root, _, files in os.walk(directory):
        # Iterate through the files in the current directory
        for file in files:
            # Check if the file has a supported image file extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Add the image file name to the list
                image_names.append(file)

    # Return the list of image file names
    return image_names


def data_processing():
    """
    Process image data from different directories.

    This function retrieves image names from specified directories, performs additional processing,
    and removes specific images based on conditions.

    Note: Replace the placeholder directory paths with the actual ones.

    Returns:
    - None
    """
    # Define directory paths
    directory_path_th0 = r"data\images_th0"  # Replace with the actual directory path
    directory_path_th1 = r"data\images_th1"  # Replace with the actual directory path
    directory_path_depth = (
        r"data\images_depth"  # Replace with the actual directory path
    )
    directory_path_rgb = r"data\images_rgb"  # Replace with the actual directory path

    # Get image names from respective directories
    image_names_th0 = get_image_names(directory_path_th0)
    image_names_th1 = get_image_names(directory_path_th1)
    image_names_depth = get_image_names(directory_path_depth)
    image_names_rgb = get_image_names(directory_path_rgb)

    # Process images in the 'images_th1' directory
    for name in image_names_th1:
        # Generate corresponding RGB image name
        rgb_name = name[0:7] + "rgb8" + name[15:]

        # Print RGB image name and check if it exists in the 'images_rgb' directory
        print(rgb_name)
        print(search_image_by_name(directory_path_rgb, rgb_name))

        # If the RGB image does not exist, remove associated images from other directories
        if not search_image_by_name(directory_path_rgb, rgb_name):
            th0_name = name[0:7] + "thermal0" + name[15:]
            depth_name = name[0:7] + "depth16" + name[15:]

            # Remove images from respective directories
            remove_image_by_name(directory_path_th0, th0_name)
            remove_image_by_name(directory_path_th1, name)
            remove_image_by_name(directory_path_depth, depth_name)

            # Print the names of removed images
            print(th0_name, depth_name, name)


def manual_calibration():
    """
    Perform manual calibration for stereo vision using chessboard images.

    Note: Replace the placeholder image paths with the actual paths in your application.

    Returns:
    - None
    """
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((10 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:10].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3d point in real-world space
    imgpoints = []  # 2d points in the image plane

    # Load images from both cameras
    img_left = cv2.imread(
        r"C:\Users\gaeta\OneDrive\Desktop\Projet_IA\HumanFinder2000\data\calibration\rgb8_460.png"
    )
    img_right = cv2.imread(
        r"C:\Users\gaeta\OneDrive\Desktop\Projet_IA\HumanFinder2000\data\calibration\rgb8_460.png",
        cv2.IMREAD_GRAYSCALE,
    )

    # Image processing for the right image
    # ... (your processing steps)

    # Lists to store calibration points for each camera
    obj_points = []
    img_points_left = []
    img_points_right = []

    # Detect chessboard corners in both images
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.equalizeHist(img_right)  # Example processing, replace as needed

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (7, 10), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (7, 10), None)

    if ret_left and ret_right:
        obj_points.append(objp)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

    # Calibrate the left camera
    ret, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        obj_points, img_points_left, gray_left.shape[::-1], None, None
    )

    # Calibrate the right camera
    ret, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        obj_points, img_points_right, gray_right.shape[::-1], None, None
    )

    # Stereo calibration
    (
        retval,
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        R,
        T,
        E,
        F,
    ) = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        gray_left.shape[::-1],
    )

    # Stereo rectification
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T
    )

    # Additional processing steps...
    # Draw corners, display images, undistort images, etc.


def display_three_images(image_path1, image_path2, image_path3):
    # Load the images from the specified paths
    image1 = resize_image(mpimg.imread(image_path1))
    image2 = resize_image(mpimg.imread(image_path2))
    image3 = resize_image(mpimg.imread(image_path3))

    # Create a figure with three subplots arranged horizontally
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display the first image in the first subplot
    axs[0].imshow(image1)
    axs[0].axis("off")  # Turn off axes
    axs[0].set_title("Image 1")  # Set the title for the subplot

    # Display the second image in the second subplot
    axs[1].imshow(image2)
    axs[1].axis("off")
    axs[1].set_title("Image 2")

    # Display the third image in the third subplot
    axs[2].imshow(image3)
    axs[2].axis("off")
    axs[2].set_title("Image 3")

    # Show the figure containing the three subplots
    plt.show()


if __name__ == "__main__":
    """
    input_images_directory = "D:\IDU5\HumanFinder2000\data\images_th0"
    output_annotated_directory = "D:/IDU5/HumanFinder2000/annotation/th0"
    annotate_images_with_resized_boxes(
        input_images_directory, output_annotated_directory
    )
    """
    solver(rgb, th0)
