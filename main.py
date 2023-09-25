from puzzle_recognizer import recognize_sudoku
from sudoku_solver.sudoku_solver import numberOfSolutions
from sudoku_solver.sudoku_solver import printSudoku

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'images/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part')
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file')
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Read the image using cv2
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    loaded_image = cv2.imread(image_path)
    if loaded_image is None:
        return jsonify(error="Failed to read the image. Please ensure it's a valid image file.")
    
    # Pass the loaded image to recognize_sudoku
    sudoku_np = recognize_sudoku(loaded_image)
    
    # Convert the numpy array to a Python list for JSON serialization
    sudoku_list = sudoku_np.tolist()
    
    # Return HTML response to display the recognized Sudoku
    return '''
    <html>
        <head>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <style>
                table {
                    border-collapse: collapse;
                }
                td {
                    border: 1px solid #000;
                    width: 2em;
                    height: 2em;
                    text-align: center;
                }
                .thick-border-right {
                    border-right-width: thick;
                }
                .thick-border-bottom {
                    border-bottom-width: thick;
                }
                .thick-border-top {
                    border-top-width: thick;
                }
                .thick-border-left {
                    border-left-width: thick;
                }
            </style>
        </head>
        <body>
            <button onclick="window.location.href='/'">Reset</button>
            Recognized Sudoku: <br>
            <div id="sudokuOutput"></div>
            <br>
            <button onclick="submitSudoku()">Solve Sudoku</button>
            <br><br>
            Solutions: <br>
            <div id="solutionsOutput"></div>
            
            <script>
                const initialSudoku = ''' + jsonify(sudoku=sudoku_list).get_data(as_text=True) + ''';
                createSudokuTable('sudokuOutput', initialSudoku.sudoku);
                
                function createSudokuTable(containerIdOrElement, sudoku) {
                    var container;
                    if (typeof containerIdOrElement === "string") {
                        // If containerIdOrElement is a string, get the element by id
                        container = document.getElementById(containerIdOrElement);
                    } else {
                        // If containerIdOrElement is a div element, use it directly
                        container = containerIdOrElement;
                    }
                    
                    const table = document.createElement('table');
                    container.appendChild(table); // Append table to the resolved container
                    
                    sudoku.forEach((row, rowIndex) => {
                        const tableRow = document.createElement('tr');
                        row.forEach((cell, cellIndex) => {
                            const tableCell = document.createElement('td');
                            tableCell.textContent = cell || '';
                            
                            if (cellIndex % 3 === 2) tableCell.classList.add('thick-border-right');
                            if (rowIndex % 3 === 2) tableCell.classList.add('thick-border-bottom');
                            if (cellIndex % 3 === 0 && cellIndex !== 0) tableCell.classList.add('thick-border-left');
                            if (rowIndex % 3 === 0 && rowIndex !== 0) tableCell.classList.add('thick-border-top');
                            if (cellIndex === 0) tableCell.classList.add('thick-border-left');
                            if (rowIndex === 0) tableCell.classList.add('thick-border-top');
                            
                            tableRow.appendChild(tableCell);
                        });
                        table.appendChild(tableRow);
                    });
                }
                
                function submitSudoku() {
                    // Make AJAX request to solve the Sudoku
                    $.ajax({
                        url: "/solve",
                        type: "POST",
                        data: JSON.stringify({ sudoku: initialSudoku.sudoku.flat() }),
                        contentType: "application/json",
                        dataType: "json",
                        success: function(data) {
                            if (data.error) {
                                alert(data.error);
                                return;
                            }
                            
                            const solutionsDiv = document.getElementById('solutionsOutput');
                            solutionsDiv.innerHTML = ''; // Clear previous solutions
                            
                            data.solutions.forEach((solution, index) => {
                                const solutionDiv = document.createElement('div'); // Create a new div for each solution
                                const heading = document.createElement('p');
                                heading.textContent = "Solution " + (index + 1) + ":";
                                solutionDiv.appendChild(heading); // Append heading to the new div
                                
                                createSudokuTable(solutionDiv, solution); // Pass the new div as the container
                                solutionsDiv.appendChild(solutionDiv); // Append the new div to solutionsOutput
                            });
                        }
                    });
                }
            </script>
        </body>
    </html>
    '''





@app.route('/solve', methods=['POST'])
def solve_sudoku():
    data = request.get_json(force=True)
    print(data)
    flattened_sudoku = data.get("sudoku")

    if not flattened_sudoku or len(flattened_sudoku) != 81:
        return jsonify(error="Invalid Sudoku data.")

    num_solutions, solutions = numberOfSolutions(flattened_sudoku)

    # Convert the flattened solutions back to 2D arrays (9x9)
    solutions_2D = [[solution[i:i+9] for i in range(0, 81, 9)] for solution in solutions]


    return jsonify(num_solutions=num_solutions, solutions=solutions_2D)

@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h2>Upload a Sudoku Image</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload and Recognize">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
