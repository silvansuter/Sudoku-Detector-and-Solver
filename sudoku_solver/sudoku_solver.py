import numpy as np
from ortools.sat.python import cp_model

def sudokuSolver(inputSudoku, othersolution_support=[]):
    """
    Solves a given Sudoku using Constraint Programming.
    
    Args:
        inputSudoku (list): A list representation of the Sudoku to solve.
        othersolution_support (list, optional): A list of previously computed solutions to avoid.
        
    Returns:
        outputSudoku (list): A list representation of the solved Sudoku.
        support (list): A list representation of the support values for the solution.
    """
    
    # Create a CP model.
    model = cp_model.CpModel()

    # Define the variables for each cell and number
    variables = [[[model.NewBoolVar(f'x{i}{j}{k}') for k in range(9)] for j in range(9)] for i in range(9)]

    # Constraint: Each cell should have one number
    for i in range(9):
        for j in range(9):
            model.Add(sum(variables[i][j]) == 1)

    # Constraints for rows and columns
    for k in range(9):
        for i in range(9):
            model.Add(sum(variables[i][j][k] for j in range(9)) == 1)
            model.Add(sum(variables[j][i][k] for j in range(9)) == 1)

    # Constraints for 3x3 boxes
    for l in [0, 3, 6]:
        for r in [0, 3, 6]:
            for k in range(9):
                model.Add(sum(variables[l + i][r + j][k] for i in range(3) for j in range(3)) == 1)

    # Place the known numbers from the input Sudoku
    for i in range(9):
        for j in range(9):
            k = inputSudoku[9*i+j] - 1
            if k >= 0:
                model.Add(variables[i][j][k] == 1)
    
    # Avoid previously computed solutions
    for other_solution in othersolution_support:
        dot_product = [variables[i][j][k] * other_solution[81*i+9*j+k] for i in range(9) for j in range(9) for k in range(9)]
        model.Add(sum(dot_product) < 81)

    model.Minimize(0)

    # Solve the CP model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3.0  # set a timeout
    status = solver.Solve(model)

    # If no solution found, return None
    if status != cp_model.OPTIMAL:
        return None, None

    # Extract solution and support
    outputSudoku = [0] * len(inputSudoku) # This array will contain the (flattened) solution of the sudoku
    support = [0] * (9**3) # This array will contain the one-hot vectors for each of the 9^2 entries of the sudoku, again flattened

    for i in range(9):
        for j in range(9):
            for k in range(9):
                if solver.Value(variables[i][j][k]) == 1:
                    outputSudoku[9*i+j] = k + 1
                    support[81*i+9*j+k] = 1

    return outputSudoku, support

def ComputeAllSolutions(inputSudoku, maxNumber=10):
    """
    Computes the number of solutions for a given Sudoku puzzle up to a specified limit.

    Args:
        inputSudoku (list or 2D array): Input Sudoku in list form or as a 2D array.
        maxNumber (int, optional): The maximum number of solutions to compute. Defaults to 10.

    Returns:
        tuple: 
            int: The number of solutions found.
            list: A list of solutions, each in list form.
    """
    # Initialize lists to store solutions and their supports
    sols = []
    supps = []

    # Flatten the 2D input array to a 1D list and convert elements to integers
    inputSudoku = list(np.array(inputSudoku).flatten().astype(int))

    # Try to solve the Sudoku using the solver function
    sol, supp = sudokuSolver(inputSudoku)

    # As long as solutions are found, add them to the list and avoid those in subsequent solves
    while sol is not None:
        sols.append(sol)
        supps.append(supp)

        # Stop if maximum number of solutions has been reached
        if len(sols) == maxNumber:
            break

        # Try finding another solution while avoiding previously found ones
        sol, supp = sudokuSolver(inputSudoku, supps)

    # Return the number of solutions found and the list of solutions
    return len(sols), sols