import cv2
import sys
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
from os.path import abspath

########################################################################################################################
# This function applies gaussian blur and adaptive threshold to an image.
#  x is the threshold set for the blur which has been set to 2 after multiple test.
# It returns the image after processing.


def pre_process(x, image):

    image = cv2.GaussianBlur(image, (2 * x + 1, 2 * x + 1), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.bitwise_not(image)

    return image

########################################################################################################################


def main():

    model_dir = abspath('../model')

    # load json and create a model based on the json file
    json_file = open(model_dir+'/mnist.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_dir+"/mnist.h5")

    while True:

        # Getting the image which has to be predicted from the data folder
        img_path = input("Please, give me the path of the picture you want to predict : :")
        image = cv2.imread(abspath(img_path))
        original = image.copy()
        predicted = original.copy()

        # Little lambda function which attributes 0 to even and 1 to odd
        even = lambda x: "Even" if x == 1 else "Odd"

        # Convert the image to greyscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))

        # Processing of the image tanks to the function
        image = pre_process(2, image)

        # Prediction based on the model used and load previously
        x = img_to_array(image)
        x = x/255
        x = np.expand_dims(x, axis=0)
        predications = model.predict_classes(x)
        # prob = model.predict_proba(x) => we can also have the probability associated to each predicate

        # Display of the class of the number in the terminal : odd or even
        print(f"The class of your number is :  {even(predications[0])} ")

        # Display of the result on an image
        pos = [image.shape[0]+5, image.shape[0]+100]
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_size = 0.3
        font_color = (255, 0, 0)
        cv2.putText(predicted, f"It is : {even(predications[0])} ", tuple(pos), font, font_size,
                    font_color, lineType=cv2.LINE_AA)
        cv2.imshow('pre-processing', original)
        cv2.imshow('post-processing', image)
        cv2.imshow('Predicted number', predicted)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()
            sys.exit()


if __name__ == '__main__':
    main()
