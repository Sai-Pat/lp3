
import java.util.Scanner;

class NQueens {

    static boolean isSafe(int board[][],
                          int row, int col, int n) {

        // Check upper column
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 1)
                return false;
        }

        // Check upper left diagonal
        int i = row - 1;
        int j = col - 1;

        while (i >= 0 && j >= 0) {
            if (board[i][j] == 1)
                return false;

            i--;
            j--;
        }

        // Check upper right diagonal
        i = row - 1;
        j = col + 1;

        while (i >= 0 && j < n) {
            if (board[i][j] == 1)
                return false;

            i--;
            j++;
        }

        return true;
    }

    static boolean solveNQueens(int board[][],
                                int row, int n) {

        // All queens are placed
        if (row == n)
            return true;

        // Try each column
        for (int col = 0; col < n; col++) {

            if (isSafe(board, row, col, n)) {

                board[row][col] = 1;

                if (solveNQueens(board, row + 1, n))
                    return true;

                // Backtracking
                board[row][col] = 0;
            }
        }

        return false;
    }

    static void printSolution(int board[][], int n) {

        System.out.println("\nN-Queens Matrix:");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String args[]) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the value of N: ");
        int n = sc.nextInt();

        if (n <= 0) {
            System.out.println("Enter a positive value of N.");
            sc.close();
            return;
        }

        int board[][] = new int[n][n];

        // Place first queen
        board[0][0] = 1;

        if (solveNQueens(board, 1, n)) {
            printSolution(board, n);
        } else {
            System.out.println("Solution does not exist.");
        }

        sc.close();
    }
}
