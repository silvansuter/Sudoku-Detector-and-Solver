# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from skimage import measure
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def find_puzzle(image, debug=False):
    """Locate the Sudoku puzzle in the provided image and return a warped grayscale version of it. The approach is taken from https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/.

    Args:
        image (numpy.ndarray): The input image containing the Sudoku puzzle.
        debug (bool, optional): If True, visualizes each step of the image processing pipeline. Defaults to False.

    Raises:
        Exception: If the Sudoku puzzle outline could not be found.

    Returns:
        tuple: A 2-tuple containing the RGB and grayscale warped images of the located puzzle.
    """
    
    # Convert the image to grayscale and apply a slight Gaussian blur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3) # To reduce noise.
    
    # Apply adaptive thresholding to highlight main features and then invert the threshold map for contour detection
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_threshold = cv2.bitwise_not(thresholded_image)
    
    if debug:
        cv2.imshow("Thresholded Puzzle Image", inverted_threshold)
        cv2.waitKey(0)

    # Find contours in the thresholded image, and sort them by area in descending order
    contours = cv2.findContours(inverted_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Loop through the sorted contours and identify the largest one that approximates to a rectangle
    puzzle_contour = None
    for contour in sorted_contours:
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)

        # If the approximated contour has four points, it could be the puzzle's outline
        if len(approximated_contour) == 4:
            puzzle_contour = approximated_contour
            break

    if puzzle_contour is None:
        raise Exception("Could not find the Sudoku puzzle outline. Debug the thresholding and contour detection steps.")

    if debug:
        cv2.drawContours(image, [puzzle_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Detected Puzzle Outline", image)
        cv2.waitKey(0)

    # Warp the detected puzzle area to get a top-down view
    warped_rgb = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped_gray = four_point_transform(gray_image, puzzle_contour.reshape(4, 2))

    if debug:
        cv2.imshow("Warped Puzzle", warped_rgb)
        cv2.waitKey(0)

    return (warped_rgb, warped_gray)

def remove_boundary_segments(image):
    """
    Remove segments from the image that only touch the boundary without extending to the middle.

    This function is designed to filter out unwanted segments, often residues from 
    grid lines in sudoku images, while retaining valid segments, like numbers 
    that might touch the boundary but are still centered in the cell.

    Args:
        image (numpy.ndarray): Binary image where segments are represented by 
                               non-zero values and the background is zero.

    Returns:
        numpy.ndarray: Processed binary image with unwanted boundary-only segments removed.
    """
    
    labeled_image = measure.label(image)
    height, width = image.shape

    # Define a margin to check if a segment extends into the middle
    margin = int(0.2 * height)  # For example, 20% of the height and width. Adjust as necessary.
    middle_area = (slice(margin, -margin), slice(margin, -margin))

    for label in np.unique(labeled_image):
        region = (labeled_image == label)
        
        touches_boundary = (
            np.any(region[0, :]) or  # Top boundary
            np.any(region[-1, :]) or  # Bottom boundary
            np.any(region[:, 0]) or  # Left boundary
            np.any(region[:, -1])  # Right boundary
        )
        
        extends_into_middle = np.any(region[middle_area])
        
        # If the segment touches the boundary and doesn't extend into the middle, remove it
        if touches_boundary and not extends_into_middle:
            image[labeled_image == label] = 0
    
    return image


#might work, see https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/

def compute_digit(image, digit_model):
    """
    Compute the digit present in an image using a deep learning model.

    Args:
        image: Input image containing a single digit.
        digit_model: Trained model for digit recognition.

    Returns:
        int: Recognized digit. Returns 0 if no digit is recognized.
    """
    # Resize the image
    image = cv2.resize(image, (28, 28))
    
    # Preprocess: blur and then apply adaptive thresholding
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to reduce noise
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Retain only the segments that touch the boundary
    thresh = remove_boundary_segments(thresh)
    
    # Find the largest contour in the cell
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Validate contour based on aspect ratio and bounding box position
    aspect_ratio = w / h
    if aspect_ratio < 0.2 or aspect_ratio > 5:
        return 0
    if w * h < 0.03 * 28 * 28: # Size constraints
        return 0
    if x > 12 or y > 12 or (x + w) < 16 or (y + h) <16: # Position constraints
        return 0
    
    # Mask the digit
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Prepare the image for the model and make the prediction
    digit = np.asarray(digit, dtype=np.float32) / 255.0
    digit = np.expand_dims(digit, axis=[0, -1])
    return np.argmax(digit_model.predict(digit, verbose=False))


def compile_model():
    """
    Compile and return a trained digit recognition model.

    Returns:
        keras.models.Model: Compiled and trained digit recognition model.
    """
    # Define the model architecture
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile and load pretrained weights
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('model_training/digit_recognizer.h5')
    
    # Return compiled model with trained weights
    return model

def recognize_sudoku(image):
    """
    Recognizes and extracts the digits of a Sudoku puzzle from an image.

    Args:
        image (numpy.ndarray): The input image containing the Sudoku puzzle as a color image.

    Returns:
        numpy.ndarray: A 9x9 matrix (2D numpy array) representing the recognized digits of the Sudoku puzzle.
                       A value of 0 indicates an empty cell.
    """

    # Load the pre-trained digit recognition model
    digit_model = compile_model()

    # Find the Sudoku puzzle and warp it to get a top-down view
    (puzzle, warped) = find_puzzle(image)

    # Get the dimensions of the warped grayscale image
    l, h = warped.shape

    # Initialize a 9x9 matrix filled with zeros to store the recognized digits
    A = np.zeros((9,9))

    # Iterate through each of the 9 rows
    for i in range(9):
        # Iterate through each of the 9 columns
        for j in range(9):
            # Extract the individual cell from the Sudoku grid based on the current row and column
            # The warping process ensures the cells are equally distributed, so we can easily slice them
            puzzle_ij = warped[int(l/9*i):int(l/9*(i+1)), int(h/9*j):int(h/9*(j+1))]
            
            # Recognize the digit in the extracted cell using the digit recognition model
            # and store the result in the matrix
            A[i,j] = compute_digit(puzzle_ij, digit_model)

    # Return the recognized Sudoku puzzle as a 9x9 matrix
    return A
