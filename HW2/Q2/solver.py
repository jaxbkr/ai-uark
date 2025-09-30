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

def find_empty(board) -> tuple[int, int]:
    for i in range(len(board)):
        try:
            return i, board[i].index(0), 
        except:
            continue
    return -1, -1


def is_valid(board, guess, pos) -> bool:
    row, col = pos

    # Check row
    if guess in board[row]:
        return False

    # Check column
    for r in range(9):
        if board[r][col] == guess:
            return False

    # Check grid
    grid_row, grid_col = (row // 3) * 3, (col // 3) * 3
    
    for r in range(grid_row, grid_row + 3):
        for c in range(grid_col, grid_col + 3):
            if board[r][c] == guess:
                return False

    return True

def solve_sudoku(board) -> bool:
    row, col = find_empty(board)

    if row == -1 and col == -1:
        return True

    for guess in range(1, 10):
        if is_valid(board, guess, (row, col)):
            board[row][col] = guess

            if solve_sudoku(board):
                return True

            # backtrack
            board[row][col] = 0

    return False
    

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

another_sudoku_board = [
    [2,6,0,0,0,8,0,0,0],
    [0,8,0,0,0,9,6,0,0],
    [0,0,0,0,5,0,0,0,0],
    [0,9,4,3,0,0,5,0,0],
    [0,0,2,0,7,0,0,0,0],
    [0,5,0,0,0,0,8,0,4],
    [0,3,5,8,0,0,0,0,0],
    [0,0,0,0,0,0,3,0,1],
    [7,0,0,0,6,0,0,0,0]
]

print("Initial Sudoku Board:")
print_board(sudoku_board)

if solve_sudoku(sudoku_board):
    print("Solved Sudoku Board:")
    print_board(sudoku_board)
else:
    print("No solution exists.")