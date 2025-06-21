# AI-Week4-assignment
## Task 1: AI-Powered Code Completion
### **Manual Implementation**
```python
def get_sort_key(item, key):
    return item[key]
def sort_dicts_manual(data, key):
    def key_function(item):
        return get_sort_key(item, key)
    return sorted(data, key=key_function)
```

### **AI-Generated Implementation**
### CO-PILOT
```python
def sort_dicts_by_key(data, sort_key, reverse=False):
    return sorted(data, key=lambda x: x.get(sort_key, None), reverse=reverse)
```

### **Usage Example**
```python
data = [
    {'name': 'Alice', 'age': 20},
    {'name': 'Ashley', 'age': 29},
    {'name': 'Akwana', 'age': 26}
]

sorted_data = sort_dicts_manual(data, 'age')

print("Sorted by age: ")
for person in sorted_data:
    print(person)
```
# Efficiency Analysis
The AI-generated implementation is more efficient and preferable for several reasons:
1. Conciseness: The AI version accomplishes the same task in a single line of code within the function, while the manual version uses three functions(including a nested one).

2. Built-in method: The AI code uses ```dict.get()``` which is more robust as it handles missing keys gracefully (returning None of raising KeyError), while the manual version will fail if a key is missing.

3. Additional Functionality: The AI implementation includes a ```reverse``` parameter for descending sort, providing more functionality without significant overhead.

4. Performance: Both implementations have O(n log n) time complexity for sorting, but the AI version has slightly better constant factors by:
    - Avoiding the nested function calls (manual version calls ```key_function``` → ```get_sort_key```)
    - Using a lambda directly in the sorted() call
5. Readability: While both are readable, the AI version is more pythonic by using standard library features(```dict.get()``` and lambda) effectively.

The only potential advantage of the manual version is slightly better debugging capability due to named functions, but this is negligible in production code. Overall, the AI-generated version is superior in efficiency, robustness, and functionality.