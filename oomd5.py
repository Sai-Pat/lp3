
# N-Queens Problem using Backtracking

def print_solution(board, n):
    print("\nN-Queens Matrix:")
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=" ")
        print()


def is_safe(board, row, col, n):

    # Check upper column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    # Check upper right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True


def solve_n_queens(board, row, n):

    # All queens are placed
    if row == n:
        return True

    # Try each column
    for col in range(n):

        if is_safe(board, row, col, n):

            board[row][col] = 1

            if solve_n_queens(board, row + 1, n):
                return True

            # Backtracking
            board[row][col] = 0

    return False


# Main Program
n = int(input("Enter the value of N: "))

board = [[0 for _ in range(n)] for _ in range(n)]

# Place first queen at first row and first column
if n > 0:
    board[0][0] = 1

    if solve_n_queens(board, 1, n):
        print_solution(board, n)
    else:
        print("Solution does not exist.")

else:
    print("Enter a positive value of N.")
