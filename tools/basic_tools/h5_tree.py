import h5py

def print_h5_tree(obj, indent=0):
    """Рекурсивно печатает структуру HDF5 файла"""
    for key in obj.keys():
        print('  ' * indent + f'├─ {key}')
        if isinstance(obj[key], h5py.Group):
            print_h5_tree(obj[key], indent + 1)