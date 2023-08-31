from puzzle_recognizer import recognize_sudoku
from sudoku_solver.sudoku_solver import numberOfSolutions
from sudoku_solver.sudoku_solver import printSudoku

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
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
        </head>
        <body>
            Recognized Sudoku: <br>
            <pre id="sudokuOutput"></pre>
            <script>
                const sudokuData = ''' + jsonify(sudoku=sudoku_list).get_data(as_text=True) + ''';
                document.getElementById('sudokuOutput').innerText = JSON.stringify(sudokuData);
            </script>

            <button onclick="submitSudoku()">Solve Sudoku</button>
            <br><br>
            Solutions: <br>
            <pre id="solutionsOutput"></pre>
            <script>
                function submitSudoku() {
                    // Convert the displayed Sudoku into a flattened array
                    const sudokuString = document.getElementById('sudokuOutput').innerText;
                    const sudoku2D = JSON.parse(sudokuString).sudoku;
                    const flattenedSudoku = [].concat.apply([], sudoku2D);
                    // Make AJAX request to solve the Sudoku
                    console.log(flattenedSudoku);
                    $.ajax({
                        url: "/solve",
                        type: "POST",
                        data: JSON.stringify({ sudoku: flattenedSudoku }),  // Stringify the data
                        contentType: "application/json",  // Set the content type to JSON
                        dataType: "json",
                        success: function(data) {
                            if (data.error) {
                                alert(data.error);
                                return;
                            }
                            // Display the solution(s) on the webpage
                            let solutionsString = "";
                            data.solutions.forEach((solution, index) => {
                                solutionsString += "Solution " + (index + 1) + ":\\n";
                                solution.forEach(row => {
                                    solutionsString += row.join(' ') + "\\n";
                                });
                                solutionsString += "\\n";
                            });
                            document.getElementById('solutionsOutput').innerText = solutionsString;
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
