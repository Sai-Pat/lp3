
# Gradient Descent Algorithm to Find Local Minimum
# Function: y = (x + 3)^2

# Define the function
def function(x):
    return (x + 3) ** 2

# Define the derivative
def derivative(x):
    return 2 * (x + 3)

# Initialize values
x = 2
learning_rate = 0.1
iterations = 30

print("Gradient Descent Algorithm")
print("--------------------------")

# Perform Gradient Descent
for i in range(iterations):

    gradient = derivative(x)

    # Update x value
    x = x - learning_rate * gradient

    print("Iteration:", i + 1,
          "x =", round(x, 6),
          "y =", round(function(x), 6))

# Display final result
print("\nFinal Results:")
print("Local Minimum x =", x)
print("Minimum Value y =", function(x))
