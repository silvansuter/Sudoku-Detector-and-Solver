# Sudoku Solver and Recognizer

## Description
This project utilizes image processing and constraint programming (integer optimization) to recognize and solve Sudoku puzzles from images, allowing users to upload an image of a Sudoku puzzle and receive the solved puzzle as output. The recognizer understands both hand-written and printed numbers. If the Sudoku has multiple solutions, the solver is able to find all of them.

## Features
- **Sudoku Recognition**: Extracts Sudoku puzzles from images of Sudokus that contain both handwritten or printed digits.
- **Sudoku Solving**: Solves the recognized Sudoku puzzles using constraint programming.
- **Web Interface**: Provides a user-friendly interface to interact with the application.

## Installation & Setup
```sh
# Clone the repository to your local machine.
git clone <repository-url>

# Install the required packages.
pip install -r requirements.txt

# Run the application.
python main.py
```
Then navigate to `http://127.0.0.1:5000/` in your web browser to use the application.

## Usage
1. **Upload Page**: Users are greeted with an upload page, allowing them to upload an image of a Sudoku puzzle.
2. **Recognition & Solution**: After uploading, the recognized puzzle is displayed, and users can request solutions for the recognized Sudoku.

Pressing the 'reset' button, allows to upload a new image.

## File Structure
- `main.py`: Main file containing the Flask application and routes.
- `demo.ipynb`: Jupyter Notebook serving as a demonstration of the recognizing and solving functions.
- `puzzle_recognizer.py`: File containing functions and logic related to recognizing Sudoku puzzles from images.
- `sudoku_solver.py`: File containing Sudoku solving logic, using constraint programming and Google OR-Tools, and related functions.
- `representing_sudoku.py`: File containing functions for visualizing Sudokus in Python
- `digit_recognizer.py`: File for training CNN on images of hand-written and printed digits.
- `loader.py`: File for loading images of printed digits.

## Technologies Used
- **OpenCV**: For image processing tasks.
- **Keras**: For training a CNN for recognizing handwritten and printed digits.
- **Google OR-Tools**: For solving Sudoku using constraint programming.
- **Flask**: For creating the web application.
- **JavaScript & jQuery**: For dynamic client-side scripting.

## Sudoku Recognition Approach
In a first step, the Sudoku image is located in the image. The image is then warped so that the Sudoku lies flat. The cells of the Sudoku can now be located evenly distributed in the 9x9 grid. After cleaning these cells from noise, a convolutional nerual network is used to predict the number in each cell (or 0, if the cell is empty).

The digit recognition is handled by a Convolutional Neural Network (CNN) created using TensorFlow's Keras. The model structure is as follows:
```python
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
```

## Sudoku Solving Approach
The Sudoku solving is accomplished using constraint programming with Google OR-Tools. The code is structured to form constraints such that each row, column, and 3x3 box has unique numbers from 1 to 9. It also accommodates any predefined numbers in the input Sudoku and finds solutions that adhere to all the constraints. Additionally, the solver is capable of finding multiple solutions, if they exist, by avoiding previously found solutions in subsequent solving attempts. The solver’s function and constraints are defined in `sudoku_solver.py`.

Here’s a snippet demonstrating the solving approach:
```python
from ortools.sat.python import cp_model

def sudokuSolver(inputSudoku, othersolution_support=[]):
    model = cp_model.CpModel()
    # Define variables, add constraints and solve the model.
    # ...
```

## Demonstration
`demo.ipynb` serves as a demonstration notebook, illustrating the capabilities of the Sudoku recognizing and solving functionalities.

## References
The contents of the 'printed_digits_dataset'-folder are from https://github.com/kaydee0502/printed-digits-dataset. See also the license in that folder.

The approach of localizing the Sudoku and warping is taken from https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/. The preparation of digits is loosely based also on this post.

The function for plotting a Sudoku with Python is taken from https://codegolf.stackexchange.com/questions/126930/draw-a-sudoku-board-using-line-drawing-characters.

## Licensing
This project is licensed under the MIT License. This allows others to use, modify, and distribute this software without restriction.

For full details, please see the LICENSE file in the repository.
