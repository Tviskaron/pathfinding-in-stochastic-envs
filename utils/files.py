from pathlib import Path


def select_free_dir_name(rood_dir, max_id=100000):
    for cnt in range(1, max_id):
        free_folder = f"{cnt}".zfill(4)
        full_path = Path(rood_dir) / Path(free_folder)
        if not full_path.exists():
            return free_folder
    raise KeyError(f"Can't select a folder in {max_id} attempts")
