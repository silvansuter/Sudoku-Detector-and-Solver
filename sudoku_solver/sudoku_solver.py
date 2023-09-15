import numpy as np
from ortools.sat.python import cp_model

def printSudoku(sudoku):
    # compact Sudoku printing function
    # taken from https://codegolf.stackexchange.com/questions/126930/
    #    draw-a-sudoku-board-using-line-drawing-characters
    q = lambda x,y:x+y+x+y+x
    r = lambda a,b,c,d,e:a+q(q(b*3,c),d)+e+"\n"
    print(((r(*"╔═╤╦╗") + q(q("║ %d │ %d │ %d "*3 + "║\n",r(*"╟─┼╫╢")), r(*"╠═╪╬╣")) +
            r(*"╚═╧╩╝")) % tuple(sudoku)).replace(*"0 "))

def computeAllSolutions(inputSudoku, othersolution_support=[]):
    # Create a CP model.
    model = cp_model.CpModel()

    # Variables
    variables = [[[model.NewBoolVar(f'x{i}{j}{k}')
                   for k in range(9)]
                  for j in range(9)]
                 for i in range(9)]

    # Constraints
    # Each cell should have one number.
    for i in range(9):
        for j in range(9):
            model.Add(sum(variables[i][j]) == 1)

    # Rows and columns constraints
    for k in range(9):
        for i in range(9):
            model.Add(sum(variables[i][j][k] for j in range(9)) == 1)
            model.Add(sum(variables[j][i][k] for j in range(9)) == 1)

    # Box constraints
    for l in [0, 3, 6]:
        for r in [0, 3, 6]:
            for k in range(9):
                model.Add(sum(variables[l + i][r + j][k] for i in range(3) for j in range(3)) == 1)

    # Fixed numbers from the input sudoku
    for i in range(9):
        for j in range(9):
            k = inputSudoku[9*i+j] - 1
            if k >= 0:
                model.Add(variables[i][j][k] == 1)

    # Use the support of other solutions to form additional constraints. We do not want our solution to be equal to an already computed one.
    for other_solution in othersolution_support:
        dot_product = []
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    dot_product.append(variables[i][j][k] * other_solution[81*i+9*j+k])
        model.Add(sum(dot_product) < 81)

    # We solve for any feasible solution of the integer program
    model.Minimize(0)

    # Solve the CP
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # If no solution exists, return None
    if status != cp_model.OPTIMAL:
        return None, None

    # Otherwise return the solution of the sudoku, as well as the support
    outputSudoku = [0] * len(inputSudoku) # This array will contain the (flattened) solution of the sudoku
    support = [0] * (9**3) # This array will contain the one-hot vectors for each of the 9^2 entries of the sudoku, again flattened

    for i in range(9):
        for j in range(9):
            for k in range(9):
                if solver.Value(variables[i][j][k]) == 1:
                    outputSudoku[9*i+j] = k + 1
                    support[81*i+9*j+k] = 1

    return outputSudoku, support

# This function creates a string representation of a given python-mip model
def model_to_str(model):
    s = f'{model.name}:\n{model.sense}\n{model.objective}\n'
    if model.constrs:
        s += 'SUBJECT TO\n' + '\n'.join(map(str, model.constrs)) + '\n'
    s += 'VARIABLES\n' + '\n'.join(f'{v.lb} <= {v.name} <= {v.ub} {v.var_type}' for v in model.vars)
    return s

def numberOfSolutions(inputSudoku, maxNumber = 10):
    # INPUT:    inputSudoku: a sudoku which is either in list form (first row, then second row, ...) or as 2D array
    #           maxNumber: the maximum number of solutions that get computed
    # OUTPUT:   len(sols): the number of solutions
    #           sols: a list of solutions, each in list form
    
    # Create empty arrays for saving the solutions
    sols = []
    supps = []
    
    # Bring the Sudoku into the desired form
    inputSudoku = list(np.array(inputSudoku).flatten().astype(int))
    
    # Compute a first solution
    sol, supp = computeAllSolutions(inputSudoku)
    
    # Compute a next solution, as long as there is was a new solution for a sudoku
    while sol != None:
        sols.append(sol)
        supps.append(supp)
        # If we computed the maximum number of solutions, we leave the loop early
        if len(sols) == maxNumber:
            break
        sol, supp = computeAllSolutions(inputSudoku, supps)
    
    # Return the number of solutions, as well as the solutions
    return len(sols), sols