def printSudoku(sudoku):
    """
    Prints the Sudoku in a formatted visual representation.
    Taken from https://codegolf.stackexchange.com/questions/126930/draw-a-sudoku-board-using-line-drawing-characters.
    
    Args:
        sudoku (list): A list representation of the Sudoku (first row, then second row, ...).
    """
    q = lambda x,y:x+y+x+y+x
    r = lambda a,b,c,d,e:a+q(q(b*3,c),d)+e+"\n"
    print(((r(*"╔═╤╦╗") + q(q("║ %d │ %d │ %d "*3 + "║\n",r(*"╟─┼╫╢")), r(*"╠═╪╬╣")) +
            r(*"╚═╧╩╝")) % tuple(sudoku)).replace(*"0 "))
    
def model_to_str(model):
    """
    Create a string representation of a given python-mip model.
    
    Args:
        model (Model): python-mip model.
        
    Returns:
        str: String representation of the model.
    """
    s = f'{model.name}:\n{model.sense}\n{model.objective}\n'
    s += 'SUBJECT TO\n' + '\n'.join(map(str, model.constrs)) + '\n' if model.constrs else ''
    s += 'VARIABLES\n' + '\n'.join(f'{v.lb} <= {v.name} <= {v.ub} {v.var_type}' for v in model.vars)
    return s