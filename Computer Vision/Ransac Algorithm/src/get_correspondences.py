import cv2
import sys
import numpy as np
from os.path import abspath

img = []

########################################################################################################################
########################################################################################################################

# This function select a pattern of 6X6 on the chess board and draw all point on a picture automatically or manually.
# Moreover, it write the correspondences between 3D and 2D points on a txt file.


def extract_and_display_points_auto(image):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # We consider Z coordinate equal to zero to make things easier
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane.

    # Find the chess board corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6, 7),
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If a corners is found, add object points, image points (after refining them)
    if ret:
        
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # file where data are written
        name_data_file = '../data/created_data.txt'
        file = open(name_data_file, 'w')

        # Write the data in the file
        for i in range(len(imgpoints[0])):
            world_x = float(objpoints[0][i][0])
            world_y = float(objpoints[0][i][1])
            world_z = float(objpoints[0][i][2])
            img_x = float(imgpoints[0][i][0][0])
            img_y = float(imgpoints[0][i][0][1])
            file.write(str(world_x) + " " + str(world_y) + " " + str(world_z) + " " + str(img_x) + " " + str(img_y) +
                       "\n")
        file.close()

        # Draw and display the corners
        image = cv2.drawChessboardCorners(image, (6, 6), corners2, ret)
        cv2.imshow('extraction point automatic', image)

        return objpoints, imgpoints

########################################################################################################################
########################################################################################################################


def calibrate_camera_cv(image, objpoints, imgpoints):

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image.shape[0], image.shape[1]),
                                                       None, None)
    print("Here are the calibration parameters :\n")
    print("Camera matrix : ")
    print(mtx)
    print("\n")
    print("distortion coefficients :")
    print(dist)
    print("\n")
    print('Rotation vector :')
    print(rvecs)
    print("\n")
    print("Translation vector :")
    print(tvecs)

########################################################################################################################
########################################################################################################################


# mouse callback function
def draw_circle(event, x, y, param, flags):

    global img

    if event == cv2.EVENT_LBUTTONUP:

        # file where data are written
        name_data_file = 'created_data.txt'
        file = open(name_data_file, 'w')

        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        file.write(str(x) + "," + str(y) + "\n")

        file.close()

    cv2.imshow('select point to extract', img)

########################################################################################################################
########################################################################################################################


def main():

    global img

    # Getting the image which has to be predicted from the data folder
    img_path = input("Please, select the path of the image you want to work on  :")
    image = cv2.imread(abspath(img_path))
    original = image.copy()
    img = image.copy()

    while True:

        # Wait for user to press a key
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('original', original)

        # Extract point and display automatically them + calibration parameters using open cv function
        if key == ord('a'):
            print("Automatic selection of points")
            objpoints, imgpoints = extract_and_display_points_auto(img)
            calibrate_camera_cv(image, objpoints, imgpoints)
            print("calibration using open CV done \n")

        # Extract point by selecting them directly on the image
        elif key == ord('m'):
            print("Select points manually")
            cv2.namedWindow('select point to extract')
            cv2.setMouseCallback('select point to extract', draw_circle)

        # Saved image with points draw on it
        elif key == ord('s'):
            print("The image is saved")
            cv2.imwrite("saved_image_hw5.png", img)

        # Can be different for a none MacBook computer -> key≠27
        elif key == 27:
            cv2.destroyAllWindows()
            sys.exit()


########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    main()

########################################################################################################################
########################################################################################################################
