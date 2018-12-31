import cv2
import numpy as np
import sys

# Variables
file1 = "football.jpg"
file2 = "help.png"
image = cv2.imread(file1)
image_s = image.copy()
image_smooth = image.copy
img_grad_bis = image.copy()
image_rotated = image.copy()
help_image_temp = cv2.imread(file2)
help_image = cv2.resize(help_image_temp, (550, 400))
key = 0
index = 0
smooth_perso_state = False
smooth_CV_state = True

########################################################################################################################


def gauss_blur(x):
    global image
    if len(image.shape) != 3:
        print("error : check the number of channel")
        image_smooth = image.copy()
    else:
        image_smooth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(image_smooth, (2*x+1, 2*x+1), 0)
    cv2.imshow("homework2", blur)

########################################################################################################################


def smooth_perso(smooth):

    global image_smooth

    if len(image.shape) != 3:
        print("error : check the number of channel or reload the image by tapping i")
        image_smooth = image.copy()
    else:
        image_smooth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if smooth == 0:
        print("You have to select a bigger number")

    smooth = 3 if smooth == 1 else (5 if smooth == 2 else 7)

    for ind_x in range(smooth + 2, image_smooth.shape[1] - smooth - 2):
        for ind_y in range(smooth + 2, image_smooth.shape[0] - smooth - 2):
            temp = image_smooth[ind_y - (smooth - 1) / 2:ind_y + (smooth - 1) / 2 + 1,
                    ind_x - (smooth - 1) / 2:ind_x + (smooth - 1) / 2 + 1]
            image_smooth[ind_y][ind_x] = np.sum(temp / (smooth ** 2))

    if smooth_perso_state:
        cv2.imshow('homework2', image_smooth)

########################################################################################################################


def gradient_display(x):

    global img_grad_bis

    if len(image.shape) != 3:
        print("not enough channels to convert to grey")
        img_grad = image.copy()
    else:
        img_grad = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for pos_x in xrange(0, img_grad.shape[1], x):
        min_x = pos_x
        max_x = pos_x + x
        for pos_y in xrange(0, img_grad.shape[0], x):
            min_y = pos_y
            max_y = pos_y + x

            temp = img_grad[min_y:max_y, min_x:max_x]

            # Define center of the matrix
            center = (min_x + x / 2, min_y + x / 2)

            if temp.shape[0] != 0 and temp.shape[1] != 0:

                # Compute gradients on x_axis and y_axis
                grad_y = temp.copy()
                grad_y = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1)
                g_y = np.mean(grad_y) / 255 * 100

                grad_x = temp.copy()
                grad_x = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1)
                g_x = np.mean(grad_x) / 255 * 100

                length = (min_x + x / 2 + int(g_x), min_y + x / 2 + int(g_y))

                # display the average gradient of the window
                cv2.line(img_grad, center, length, (255, 0, 0),  thickness=2, lineType=8, shift=0)

    img_grad_bis = img_grad.copy()
    cv2.imshow('homework2', img_grad)
########################################################################################################################


def rotation(x):

    global image_rotated

    if len(image.shape) != 3:
        print("not enough channels to convert to grey")
    else:
        image_rotated = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows = image_rotated.shape[0]
    cols = image_rotated.shape[1]
    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), x, 1)
    image_rotated_f = cv2.warpAffine(image_rotated, rot_mat, (cols, rows))

    cv2.imshow('homework2', image_rotated_f)


########################################################################################################################

# Check if an image is specified in the command line and if not take an image with the camera

# if len(sys.argv) == 2:
#
#     image = sys.argv[1]
#
#
# elif len(sys.argv) < 2:
#
#     cap = cv2.VideoCapture(0)
#     ret_val, image = cap.read()
#     cap.release()
#
# else:
#     image = cv2.imread("beach.jpg")
#
# cv2.imshow("homework2", image)

########################################################################################################################

# Window and Trackbar


cv2.namedWindow('homework2', cv2.WINDOW_NORMAL)
cv2.createTrackbar('s', 'homework2', 1, 30, gauss_blur)
cv2.createTrackbar('S', 'homework2', 0, 3, smooth_perso)
cv2.createTrackbar('p', 'homework2', 250, 500, gradient_display)
cv2.createTrackbar('r', 'homework2', 0, 360, rotation)

# Keyboard input definition

while True:

    cv2.imshow('homework2', image)

    key = cv2.waitKey(0)
    print("key : %d" % key)

    # trackbar state : verification -> todo
    # Lower_case

    if key == ord('i'):
        print("Letter i input")
        image = image_s
        smooth = 0
        cv2.imshow("homework2", image)
        print ("Original image reloaded")

    elif key == ord('w'):
        print("Letter w input")
        cv2.imwrite('out.png', image)
        print ("the image has been saved with the name out.png")

    elif key == ord('g'):
        print("Letter g input")
        image = image_s
        if len(image.shape) != 3:
            print ("error : check the number of channel")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("homework2", image)
            print("the image has been converted to greyscale")

    elif key == ord('c'):
        print("Letter c input")
        image = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:

            b, g, r = cv2.split(image)
            col = r.copy()
            col[:] = 0
            image = cv2.merge((b, g, r))

            if index == 0:
                image = cv2.merge((b, col, col))
                print("blue")
            elif index == 1:
                image = cv2.merge((col, g, col))
                print("green")
            else:
                image = cv2.merge((col, col, r))
                print("red")

            index = index + 1 if index < 2 else 0

    elif key == ord('s'):
        print("Letter s input")
        image = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        smooth = cv2.getTrackbarPos("s", "homework2")
        y = 2 * smooth + 1
        image = cv2.GaussianBlur(image, (y, y), 0)
        cv2.imshow("homework2", image)
        smooth_CV_state = True
        print("the image has been converted to greyscale and smoothed")

    elif key == ord('d'):
        print("Letter d input")
        dim = (image.shape[1] // 2, image.shape[0] // 2)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
        print("Down-sample the image without smoothing done")

    elif key == ord('x'):
        print("Letter x input")
        image = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_derivative = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        scale = np.max(x_derivative) / 255
        image = (x_derivative / scale).astype(np.uint8)
        cv2.imshow("homework2", image)
        print ("x-derivative normalized done")

    elif key == ord('y'):
        print("Letter y input")
        image = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        y_derivative = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        scale = np.max(y_derivative) / 255
        image = (y_derivative/scale).astype(np.uint8)
        cv2.imshow("homework2", image)
        print ("y-derivative normalized done")

    elif key == ord('m'):
        image = image_s
        print("Letter m input")
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_derivative = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        y_derivative = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        image = np.sqrt(x_derivative ** 2, y_derivative ** 2)
        scale = np.max(image)/255
        image = (image / scale).astype(np.uint8)
        cv2.imshow("homework2", image)
        print("the magnitude of the image gradient showed")

    elif key == ord('p'):
        image = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Letter p input, please don't put the trackbar at the value 0")
        cv2.imshow("homework2", img_grad_bis)

    elif key == ord('r'):
        image_rotated = image_s
        if len(image.shape) != 3:
            print("error : check the number of channel or reload the image by tapping i")
        else:
            image_rotated = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)

        angle = cv2.getTrackbarPos("r","homework2")
        rows = image_rotated.shape[0]
        cols = image_rotated.shape[1]
        rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image_rotated_f = cv2.warpAffine(image_rotated, rot_mat, (cols, rows))

        cv2.imshow('homework2', image_rotated_f)

    elif key == ord('h'):
        print("Letter h input")
        cv2.imshow("homework2 : Help", help_image)

    elif key == 27:
        cv2.destroyAllWindows()
        print ("exit success")
        break

########################################################################################################################

    # Uppercase

    elif key == 0:  # Uppercase key

        Uppercase_key = cv2.waitKey(0)

        if Uppercase_key == ord('g'):
            image = image_s
            print("Letter G input")

            if len(image.shape) != 3:
                print("error : check the number of channel or reload the image by tapping i")
            else:
                weight = [0.114, 0.587, 0.299]  # weight to convert the image to greyscale
                m_grey = np.array(weight).reshape((1, 3))
                image = cv2.transform(image, m_grey)
                print("Converted to greyscale with my implementation")

        elif Uppercase_key == ord('s'):
            image = image_s
            print ("Letter S input")
            smooth_perso_state = True

            if len(image.shape) != 3:
                print("error : check the number of channel or reload the image by tapping i")
                image_smooth = image.copy()
            else:
                image_smooth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            smooth = cv2.getTrackbarPos("S", "homework2")

            if smooth == 0:
                image_smooth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                print("No filter applies")

            else:

                smooth = 3 if smooth == 1 else (5 if smooth == 2 else 7)

                for ind_x in range(smooth + 2, image_smooth.shape[1] - smooth - 2):
                    for ind_y in range(smooth + 2, image_smooth.shape[0] - smooth - 2):
                        temp = image_smooth[ind_y - (smooth - 1) / 2:ind_y + (smooth - 1) / 2 + 1,
                                   ind_x - (smooth - 1) / 2:ind_x + (smooth - 1) / 2 + 1]
                        image_smooth[ind_y][ind_x] = np.sum(temp / (smooth ** 2))

            print("Smooth done")
            cv2.imshow('homework2', image_smooth)

        elif Uppercase_key == ord('d'):
            print("Letter D input")
            dim = (image.shape[1] // 2, image.shape[0] // 2)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
            image = cv2.GaussianBlur(image, (5, 5), 0)  # smoothing imposed
            cv2.imshow("homework2", image)
            print("Down-sample the image without smoothing")


