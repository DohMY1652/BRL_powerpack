import sys
import os
import yaml

from typing import Optional, Any, Dict
import rospkg

def get_pkg_path(
    pkg: str
):
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(pkg)
    return pkg_path

def is_dir(folder_path: str) -> bool:
    return os.path.isdir(folder_path)

def checker(
    content: str,
    header: Optional[str] = None
):
    if header is None:
        print(f'[ INFO] checker ==> {content}')
    else:
        print(f'[ INFO] {header} ==> {content}')

def delete_lines(
    num: int
):
    for _ in range(num):
        sys.stdout.write("\033[F")  # Move the cursor up one line
        sys.stdout.write("\033[K")  # Clear the line

def color(
    line: str,
    color: str
):
    if color == 'blue':
        return '\033[94m' + f'{line}' + '\033[0m'
    elif color == 'yellow':
        return '\033[33m' + f'{line}' + '\033[0m'
    elif color == 'red':
        return '\033[91m' + f'{line}' + '\033[0m'
    else:
        return line

def save_yaml(
    folder_name: str,
    kwargs: Dict[str, Any],
    file_name: str = 'cfg.yaml'
):
    model_path = f'{get_pkg_path("pneu_rl")}/models/{folder_name}'
    with open(f'{model_path}/{file_name}', 'w') as f:
        yaml.dump(kwargs, f)

def load_yaml(
    folder_name: str
) -> Dict[str, Any]:
    model_path = f'{get_pkg_path("pneu_rl")}/models/{folder_name}'
    with open(f'{model_path}/cfg.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    return data
    
def save_yaml2(
    folder_name: str,
    kwargs: Dict[str, Any],
    file_name: str = 'cfg.yaml'
):
    model_path = f'{get_pkg_path("pneu_rl2")}/models/{folder_name}'
    with open(f'{model_path}/{file_name}', 'w') as f:
        yaml.dump(kwargs, f)

def load_yaml2(
    folder_name: str
) -> Dict[str, Any]:
    model_path = f'{get_pkg_path("pneu_rl2")}/models/{folder_name}'
    with open(f'{model_path}/cfg.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    return data

def save_kwargs(
    path: str,
    kwargs: Dict[str, Any]
) -> None:
    with open(path, "w") as f:
        yaml.dump(kwargs, f)