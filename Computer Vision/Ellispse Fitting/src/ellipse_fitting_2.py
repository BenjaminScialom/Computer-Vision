
import sys
import cv2
import numpy as np

########################################################################################################################

# Variables


name = "Ellipse Fitting"
ellipse_frame = 0
hc_frame = 0
file = "help_menu.png"
help_menu_temp = cv2.imread(file)
help_menu = cv2.resize(help_menu_temp, (550, 400))


########################################################################################################################

# This function is used to not specify the argument require


def nothing(x):
    pass

########################################################################################################################

# This function is used to change the state of the variable : false to true or true to false
# It's a means to manage the use of the different keyboard input


def modify_state(state):
    return not state

########################################################################################################################

# The return of the "findcontours" function is certain number of matrix which contains coordinates of the points.
# They need to be slplit into two different vectors which are x and y.


def decomp(contours):
    x, y = [], []
    for i in contours:
        for j in i:
            x.append(j[0])
            y.append(j[1])
    return x, y

########################################################################################################################

# This function is used to reshape the camera of the computer
# The less the percentage is the less the resolution will be good but less computing will be needed by the computer


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


########################################################################################################################

def draw_own_ellipse(a_threshold, img):

    t_img = img.copy()
    area_threshold = a_threshold

    image, contours, hierarchy = cv2.findContours(img, 1, 2)

    for x in contours:
        if len(x) > 5:

            try:
                # Use of the decomposition function to get the x-coordinates and y-coordinates in two different vectors
                contours_coordinates = decomp(x)

                # Use of the fit function which enable the program to gather which FIT the best an ellipse
                ellipse_parameters = fit(contours_coordinates)
                center, width, height, angle = ellipse_parameters
                center = (center[0].astype(int), center[1].astype(int))
                axes = (int(width), int(height))
                angle_in_deg = np.rad2deg(angle.real)
                print ("This is the list of the different parameters:", ellipse_parameters)

            except IndexError:
                print ("Issue in the draw_own_ellipse_function")
                continue

            # This is an improvement : the function draws only ellipses which have a certain size.
            # The size is chosen thanks to the track-bar
            area = np.pi*height*width
            if area >= area_threshold:
                img = cv2.ellipse(t_img, center, axes, int(angle_in_deg), 0, 360, (255, 0, 255))
                # Color Issue unresolved

    return img


########################################################################################################################

# The goal of this function is to gather points that fit the best an ellipse by minimizing the algebraic distance and
# by taking into account the fact that it should be an ellipse b^2-4ac<0 => matrix C.
# We are looking for the parameters {a,b,c,d,e,f} of the vector l which solve the equation of the ellipse

def fit(contours_coordinates):

    # From the data we split x and y coordinates to work separately on both of them
    x, y = np.asarray(contours_coordinates, dtype=float)

    #
    D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
    D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

    # This corresponds to the equation 17
    S1 = D1.T*D1
    S2 = D1.T*D2
    S3 = D2.T*D2

    # Constraint matrix
    const_matrix = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

    # This correspond to the equation equation 29
    M = const_matrix.I*(S1-S2*S3.I*S2.T)

    # Find eigenvalues and eigenvectors from the equation 28
    eval, evec = np.linalg.eig(M)

    # We check if it is really an ellipse
    check_ellipse = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
    a1 = evec[:, np.nonzero(check_ellipse.A > 0)[1]]

    # This correspond to the equation 24
    a2 = -S3.I*S2.T*a1

    # eigenvectors [a b c d f g] that correspond to ellipses
    param = np.vstack([a1, a2])

    # Eigenvectors are the coefficients of an ellipse : a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0

    a = param[0, 0]
    b = param[1, 0]/2.
    c = param[2, 0]
    d = param[3, 0]/2.
    f = param[4, 0]/2.
    g = param[5, 0]

    # Compute the center of an ellipse
    x0 = (c*d-b*f)/(b**2.-a*c)
    y0 = (a*f-b*d)/(b**2.-a*c)

    center = [x0, y0]

    # Compute the semi-axes lengths
    numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    denominator1 = (b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    denominator2 = (b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    width = np.sqrt(numerator/denominator1)
    height = np.sqrt(numerator/denominator2)

    # Compute angle of the ellipse based on the contour points
    angle = .5*np.arctan((2.*b)/(a-c))

    return center, width, height, angle


########################################################################################################################


def difference(img1, img2):

    global name

    # Compute the absolute difference between to images (frames)
    diff = cv2.absdiff(img1, img2)
    mask = diff

    # We use a threshold to keep only the pixels that have a big difference
    # The higher the threshold is the more it shows what really moves.
    th = cv2.getTrackbarPos("Threshold-difference", name)
    imask = mask > th

    # Build the matrix of the difference adjusted with the threshold
    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]

    return canvas

########################################################################################################################

# This function draw ellipses based on the opencv function "fit_ellipse".


def draw_cv_ellipse(t, img):

    t_img = img.copy()
    img = cv2.bitwise_not(img)
    ret, thresh = cv2.threshold(img, t, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        if len(cnt) > 5:
            box = cv2.fitEllipse(cnt)
            thresh = cv2.ellipse(t_img, box, (0, 0, 255))
    return thresh


########################################################################################################################


def main():

    # Open the camera and display the image continuously
    camera = cv2.VideoCapture(0)
    a, flux = camera.read()
    flux = rescale_frame(flux, 15)
    old_flux = flux

    global name
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # Track-bar declaration
    # trackbar for the precision of the ellipses
    cv2.createTrackbar('Canny Threshold 1',name,100,255,nothing)
    cv2.createTrackbar('Canny Threshold 2', name, 200, 255, nothing)

    # trackbar for the Sensibility of the movement detector
    cv2.createTrackbar('Threshold-difference', name, 127, 255, nothing)
    cv2.createTrackbar("Ellipse-size", name, 0, flux.shape[1]*flux.shape[0]//10, nothing)

    # trackbar for Threshold
    cv2.createTrackbar('Delay',name, 1, 48, nothing)

    # flags
    ellipse_cv = False
    diff = False
    edge = False
    own_ellipse = False

    # This variable is a means to adjust the delay between several frames.
    i = 0

    while True:

        # Open the camera and display the image continuously
        a, flux = camera.read()
        flux = rescale_frame(flux, 15)

        # variable "key" take the value of the key press
        key = cv2.waitKey(1) & 0xFF

        # Management of the different cases
        if key == 27:
            camera.release()
            cv2.destroyAllWindows()
            sys.exit()

        elif key == ord('d'):
            diff = modify_state(diff)

        elif key == ord('e'):
            ellipse_cv = modify_state(ellipse_cv)
            own_ellipse = False

        elif key == ord('g'):
            edge = modify_state(edge)

        elif key == ord('b'):
            own_ellipse = modify_state(own_ellipse)
            ellipse_cv = False

        elif key == ord('h'):
            print("Letter h input")
            cv2.imshow("homework3 : Help", help_menu)

        # flux of the camera converted in greyscale
        flux = cv2.cvtColor(flux, cv2.COLOR_BGR2GRAY)

        # Save the current flux
        temp_flux = flux.copy()

        # Call of the different tools according to the key press and the previous state of the variable related
        if edge:
            t1 = cv2.getTrackbarPos(trackbarname="Canny Threshold 1", winname=name)
            t2 = cv2.getTrackbarPos(trackbarname="Canny Threshold 2", winname=name)
            flux = cv2.Canny(flux, t1, t2)

        if diff:
            flux = difference(old_flux, flux)

        if ellipse_cv:
            threshold_cv = cv2.getTrackbarPos(trackbarname="Threshold CV Ellipse", winname=name)
            flux = draw_cv_ellipse(threshold_cv, flux)

        if own_ellipse:
            ellipse_size = cv2.getTrackbarPos(trackbarname="Ellipse-Size", winname=name)
            flux = draw_own_ellipse(ellipse_size, flux)

        cv2.imshow(name, flux)

        # Management of the delay between the frames
        delay = cv2.getTrackbarPos(trackbarname="Delay", winname=name) if cv2.getTrackbarPos(trackbarname="Delay", winname=name) != 0 else 1

        if i % delay == 0:
            old_flux = temp_flux

        i += 1


########################################################################################################################

if __name__ == '__main__':
    main()
