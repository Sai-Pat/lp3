
# Fractional Knapsack using Greedy Method

def fractional_knapsack(capacity, items):

    # Sort items according to profit/weight ratio
    items.sort(key=lambda x: x[0] / x[1], reverse=True)

    total_profit = 0

    for profit, weight in items:

        if capacity == 0:
            break

        if weight <= capacity:
            # Take complete item
            total_profit += profit
            capacity -= weight

        else:
            # Take fractional part of item
            fraction = capacity / weight
            total_profit += profit * fraction
            capacity = 0

    return total_profit


# Main Program
n = int(input("Enter number of items: "))

items = []

for i in range(n):
    profit = float(input("Enter profit: "))
    weight = float(input("Enter weight: "))
    items.append((profit, weight))

capacity = float(input("Enter knapsack capacity: "))

max_profit = fractional_knapsack(capacity, items)

print("Maximum Profit =", max_profit)
