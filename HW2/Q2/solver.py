def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")
    print()

def find_empty(board):
    return None

def is_valid(board, guess, pos):
    pass

def solve_sudoku(board):
    pass

sudoku_board = [
    [0,1,3,0,0,0,7,0,0],
    [0,0,0,5,2,0,4,0,0],
    [0,8,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,8,0],
    [9,0,0,0,0,0,6,0,0],
    [2,0,0,0,0,0,0,0,0],
    [0,5,0,4,0,0,0,0,0],
    [7,0,0,6,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0]
]

print("Initial Sudoku Board:")
print_board(sudoku_board)

if solve_sudoku(sudoku_board):
    print("Solved Sudoku Board:")
    print_board(sudoku_board)
else:
    print("No solution exists.")