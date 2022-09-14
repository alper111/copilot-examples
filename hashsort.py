"""Hash sort in Python"""

def hash_sort(lst):
    """Hash sort in Python"""
    hash_table = {}
    for i in lst:
        if i in hash_table:
            hash_table[i] += 1
        else:
            hash_table[i] = 1
    sorted_list = []
    for i in range(min(hash_table), max(hash_table) + 1):
        if i in hash_table:
            sorted_list.extend([i] * hash_table[i])
    return sorted_list

if __name__ == "__main__":
    import random
    lst = [random.randint(0, 100) for _ in range(100)]
    print(hash_sort(lst))

# The hash sort algorithm is a simple sorting algorithm that uses a hash table to sort a list of numbers. The algorithm is very simple and can be implemented in a few lines of code. The algorithm is not very efficient and is not recommended for large lists. The algorithm is also not stable and does not preserve the order of duplicate elements.