def get_sort_key(item, key):
    return item[key]
def sort_dicts_manual(data, key):
    def key_function(item):
        return get_sort_key(item, key)
    return sorted(data, key=key_function)
data = [
    {'name': 'Alice', 'age': 20},
    {'name': 'Ashley', 'age': 29},
    {'name': 'Akwana', 'age': 26}
]

sorted_data = sort_dicts_manual(data, 'age')

print("Sorted by age: ")
for person in sorted_data:
    print(person)