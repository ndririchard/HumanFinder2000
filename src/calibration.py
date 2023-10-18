import numpy as np
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Load images from both cameras
img_left = cv2.imread(r'C:\Users\gaeta\OneDrive\Desktop\Projet_IA\HumanFinder2000\data\calibration\rgb8_460.png')
img_right = cv2.imread(r'C:\Users\gaeta\OneDrive\Desktop\Projet_IA\HumanFinder2000\data\calibration\rgb8_460.png', cv2.IMREAD_GRAYSCALE)

# treatment for image right
gray_right = cv2.equalizeHist(img_right)
#
alpha = 1.5  # Contrast adjustment
beta = 50   # Brightness adjustment

gray_right = cv2.convertScaleAbs(img_right, alpha=alpha, beta=beta)
#
gamma = 1.5  # Adjust gamma value as needed

gray_right = cv2.LUT(img_right, np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8))
#
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_right = clahe.apply(img_right)
#
blurred = cv2.GaussianBlur(img_right, (0, 0), 3)
gray_right = cv2.addWeighted(img_right, 1.5, blurred, -0.5, 0)
cols = 7
rows = 10

# Lists to store calibration points for each camera
obj_points = []
img_points_left = []
img_points_right = []

# Detect chessboard corners in both images

gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
#gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

ret_left, corners_left = cv2.findChessboardCorners(gray_left, (cols, rows), None)
print(ret_left, corners_left)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, (cols, rows), None)
print(ret_right, corners_right)
if ret_left and ret_right:
    obj_points.append(objp)
    img_points_left.append(corners_left)
    img_points_right.append(corners_right)

# Calibrate the left camera
ret, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_points, img_points_left, gray_left.shape[::-1], None, None)

# Calibrate the right camera
ret, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(obj_points, img_points_right, gray_right.shape[::-1], None, None)

# Stereo calibration
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1])

# Stereo rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T)

corners2 = cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
imgpoints.append(corners2)
# Draw and display the corners
cv2.drawChessboardCorners(img_left, (7,10), corners2, ret_left)
cv2.imshow('img', img_left)
cv2.waitKey(50000)
# Perform undistortion and rectification on a pair of images
# left_img = cv2.imread('left_test.jpg')
# right_img = cv2.imread('right_test.jpg')
# mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
# left_img_undistorted = cv2.remap(left_img, mapx1, mapy1, cv2.INTER_LANCZOS4)

# mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)
# right_img_undistorted = cv2.remap(right_img, mapx2, mapy2, cv2.INTER_LANCZOS4)

# Now you have calibrated the two cameras and rectified the images.

