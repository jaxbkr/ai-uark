import random


def createMatrix(m,n):
    matrix = []
    # m = row
    # n = column
    for i in range(m):
        row = []
        for j in range(n):
            row.append(random.randint(0,1))
        matrix.append(row)
    return matrix

def printMatrix(matrix):
    for row in matrix:
        print(row)

def isValid(position, matrix):
    x, y = position
    if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]):
        return False
    if matrix[x][y] == 0:
        return False
    return True

def getNeighbors(position, matrix):
    x, y = position
    neighbors = []
    directions = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up
    for direction in directions:
        newX = x + direction[0]
        newY = y + direction[1]
        if isValid((newX, newY), matrix):
            neighbors.append((newX, newY))
    return neighbors


def dfs(matrix):
    start = (0,0)
    goal = (len(matrix)-1, len(matrix[0])-1)
    stack = []
    stack.append((start, [start]))
    visited = set()

    while stack:
        (current, path) = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for neighbor in getNeighbors(current, matrix):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return -1

def bfs(matrix):
    start = (0,0)
    goal = (len(matrix)-1, len(matrix[0])-1)

    queue = []
    queue.append((start, [start]))
    visited = set()

    while queue:
        (current, path) = queue.pop(0)
        if current == goal:
            return path + []
        for neighbor in getNeighbors(current, matrix):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return -1


    pass

# output format (0, 0) −> (0, 1) −> (0, 2)...

def printPath(path):
    print("Path: ")
    for i in range(len(path)):
        if i != len(path)-1:
            print(path[i], end=" -> ")
        else:
            print(path[i])

def printTracedMatrix(matrix, path):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i,j) in path:
                print("X", end=" ")
            else:
                print(matrix[i][j], end=" ")
        print()

if __name__ == "__main__":
    m = int(input("Enter number of rows: "))
    n = int(input("Enter number of columns: "))
    print("Creating matrix and selecting one with a valid path...")

    matrix = createMatrix(m,n)
    if bfs(matrix) == -1:
        while dfs(matrix) == -1:
            matrix = createMatrix(m,n)
    
    printMatrix(matrix)

    path = dfs(matrix)
    printPath(path)
    printTracedMatrix(matrix, path)

    path = bfs(matrix)
    printPath(path)
    printTracedMatrix(matrix, path)

    

