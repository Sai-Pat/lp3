
import java.util.*;
 
class QuickSortAnalysis {

    // Deterministic Partition
    static int partition(int arr[], int low, int high) {

        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {

            if (arr[j] <= pivot) {
                i++;

                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

    // Deterministic Quick Sort
    static void quickSort(int arr[], int low, int high) {

        if (low < high) {

            int pi = partition(arr, low, high);

            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    // Randomized Partition
    static int randomizedPartition(int arr[],
                                   int low, int high) {

        Random rand = new Random();

        int randomIndex = low + rand.nextInt(high - low + 1);

        int temp = arr[randomIndex];
        arr[randomIndex] = arr[high];
        arr[high] = temp;

        return partition(arr, low, high);
    }

    // Randomized Quick Sort
    static void randomizedQuickSort(int arr[],
                                    int low, int high) {

        if (low < high) {

            int pi = randomizedPartition(arr, low, high);

            randomizedQuickSort(arr, low, pi - 1);
            randomizedQuickSort(arr, pi + 1, high);
        }
    }

    public static void main(String args[]) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of elements: ");
        int n = sc.nextInt();

        int arr[] = new int[n];

        for (int i = 0; i < n; i++) {
            System.out.print("Enter element: ");
            arr[i] = sc.nextInt();
        }

        // Deterministic Quick Sort
        int a[] = arr.clone();

        long start = System.nanoTime();

        quickSort(a, 0, n - 1);

        long end = System.nanoTime();

        System.out.println("\nDeterministic Quick Sort:");
        System.out.println(Arrays.toString(a));
        System.out.println("Execution Time: " +
                           (end - start) + " ns");

        // Randomized Quick Sort
        int b[] = arr.clone();

        start = System.nanoTime();

        randomizedQuickSort(b, 0, n - 1);

        end = System.nanoTime();

        System.out.println("\nRandomized Quick Sort:");
        System.out.println(Arrays.toString(b));
        System.out.println("Execution Time: " +
                           (end - start) + " ns");

        sc.close();
    }
}
