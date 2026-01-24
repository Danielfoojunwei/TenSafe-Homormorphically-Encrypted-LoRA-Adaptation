
import os
from pathlib import Path

def get_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except OSError:
        pass
    return total

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"

def run_audit(root_dir):
    root = Path(root_dir)
    exclude_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.idea', '.vscode'}
    
    all_files = []
    dir_sizes = {}
    extensions = {'.mp4': [], '.mov': [], '.zip': [], '.tar': [], '.tgsp': [], '.onnx': [], '.pt': [], '.bin': [], '.safetensors': []}
    
    print(f"Auditing {root}...")

    # Traverse
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_dir = Path(dirpath)
        d_size = get_dir_size(current_dir)
        dir_sizes[str(current_dir)] = d_size
        
        for f in filenames:
            fp = current_dir / f
            fs = get_size(fp)
            all_files.append((str(fp), fs))
            
            ext = fp.suffix.lower()
            if ext in extensions:
                extensions[ext].append((str(fp), fs))

    # Top files
    all_files.sort(key=lambda x: x[1], reverse=True)
    print("\n=== TOP 20 LARGEST FILES ===")
    for f, s in all_files[:20]:
        print(f"{format_size(s):>10}  {f}")
        
    # Top dirs
    sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)
    print("\n=== TOP 20 LARGEST DIRECTORIES ===")
    for d, s in sorted_dirs[:20]:
        print(f"{format_size(s):>10}  {d}")
        
    # Extensions
    print("\n=== BINARY FILES ===")
    for ext, files in extensions.items():
        if files:
            print(f"\nExtension: {ext}")
            for f, s in files:
                print(f"  {format_size(s):>8} {f}")

if __name__ == "__main__":
    run_audit(".")
