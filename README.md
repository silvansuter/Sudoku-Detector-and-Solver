# Sudoku Solver and Recognizer

## Description
This project utilizes image processing and constraint programming (integer programming) to recognize and solve Sudoku puzzles from images, allowing users to upload an image of a Sudoku puzzle and receive the solved puzzle as output. The program allows to recognize both hand-written and printed numbers. If the Sudoku has multiple solutions, the solver is able to find all of them.

## Features
- **Sudoku Recognition**: Extracts Sudoku puzzles from images.
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

Pressing the 'reset' button, allows the user to enter a new image.

## File Structure
- `app.py`: Main file containing the Flask application and routes.
- `demo.ipynb`: Jupyter Notebook serving as a demonstration of the recognizing and solving functions.
- `puzzle_recognizer.py`: File containing functions and logic related to recognizing Sudoku puzzles from images.
- `sudoku_solver.py`: File containing Sudoku solving logic, using constraint programming and Google OR-Tools, and related functions.

## Technologies Used
- **OpenCV**: For image processing tasks.
- **Keras**: For training a CNN for recognizing handwritten and printed digits.
- **Google OR-Tools**: For solving Sudoku using constraint programming.
- **Flask**: For creating the web application.
- **JavaScript & jQuery**: For dynamic client-side scripting.

## Sudoku Recognition Approach
In a first step, the Sudoku image is located in the image. The image is then warped so that the Sudoku lies flat. The cells of the Sudoku can now be located evenly distributed in the 9x9 grid. After cleaning these cells from noise, a convolutional nerual network is used to predict the number in each cell (or 0, if the cell is empty).

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
The contents of 'printed_digits_dataset' are from TODO.

The approach of localizing the sudoku and warping is taken from TODO. The preparation of digits is loosely based also on TODO's approach.

The function for plotting a Sudoku with Python is taken from TODO.
