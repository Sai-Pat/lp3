
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encoding(chars, freq):
    heap = []

    # Create leaf nodes
    for char, f in zip(chars, freq):
        heapq.heappush(heap, Node(char, f))

    # Build Huffman Tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        new_node = Node(None, left.freq + right.freq)
        new_node.left = left
        new_node.right = right

        heapq.heappush(heap, new_node)

    root = heap[0]
    codes = {}

    # Generate Huffman Codes
    def generate_codes(node, code):
        if node is None:
            return

        if node.char is not None:
            codes[node.char] = code if code else "0"
            return

        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    generate_codes(root, "")

    print("\nHuffman Codes:")
    for char in chars:
        print(char, ":", codes[char])


# Main Program
n = int(input("Enter number of characters: "))

chars = []
freq = []

for i in range(n):
    ch = input("Enter character: ")
    f = int(input("Enter frequency: "))
    chars.append(ch)
    freq.append(f)

huffman_encoding(chars, freq)
