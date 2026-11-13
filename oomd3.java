
import java.util.*;

class Item {
    int profit, weight;
    double ratio;

    Item(int profit, int weight) {
        this.profit = profit;
        this.weight = weight;
        this.ratio = (double) profit / weight;
    }
}

class FractionalKnapsack {

    static double knapsack(Item[] items, int capacity) {

        // Sort items by profit/weight ratio
        Arrays.sort(items, (a, b) ->
            Double.compare(b.ratio, a.ratio)
        );

        double totalProfit = 0;

        for (Item item : items) {

            if (capacity == 0)
                break;

            if (item.weight <= capacity) {

                // Take complete item
                totalProfit += item.profit;
                capacity -= item.weight;

            } else {

                // Take fractional part
                double fraction =
                    (double) capacity / item.weight;

                totalProfit += item.profit * fraction;
                capacity = 0;
            }
        }

        return totalProfit;
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        Item[] items = new Item[n];

        for (int i = 0; i < n; i++) {

            System.out.print("Enter profit: ");
            int profit = sc.nextInt();

            System.out.print("Enter weight: ");
            int weight = sc.nextInt();

            items[i] = new Item(profit, weight);
        }

        System.out.print("Enter knapsack capacity: ");
        int capacity = sc.nextInt();

        double maxProfit = knapsack(items, capacity);

        System.out.println("Maximum Profit = " + maxProfit);

        sc.close();
    }
}
