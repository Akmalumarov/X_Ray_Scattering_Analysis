# show_tree.py
"""
Repository tree viewer
"""

from pathlib import Path

def show_tree(start_path='.', max_depth=4, exclude_dirs=None):
    """
    Display repository tree structure
    
    Args:
        start_path: root directory to scan
        max_depth: maximum folder depth to show
        exclude_dirs: list of directory patterns to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', 
                       '.DS_Store', '*.egg-info']
    
    start_path = Path(start_path)
    tree = []
    
    def add_directory(path, prefix="", depth=0):
        if depth > max_depth:
            return
            
        try:
            items = sorted(path.iterdir())
            items = [item for item in items 
                    if not any(item.name == pattern or 
                              (pattern.startswith('*') and item.name.endswith(pattern[1:]))
                              for pattern in exclude_dirs)]
            
            for index, item in enumerate(items):
                is_last = index == len(items) - 1
                
                if item.is_dir():
                    tree.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    add_directory(item, new_prefix, depth + 1)
                else:
                    tree.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                    
        except PermissionError:
            pass
    
    tree.append(f"{start_path.absolute()}/")
    add_directory(start_path)
    return "\n".join(tree)

def get_imports():
    """Get Python modules import paths"""
    imports = []
    for py_file in Path('tools').rglob("*.py"):
        if py_file.name != "__init__.py":
            rel_path = py_file.relative_to(Path('.'))
            import_path = str(rel_path.with_suffix('')).replace('/', '.')
            imports.append(import_path)
    return sorted(imports)

# For command line use
if __name__ == "__main__":
    print(show_tree())
    print("\nPython imports:")
    for imp in get_imports():
        print(f"  {imp}")